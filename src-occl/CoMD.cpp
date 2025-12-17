/// \file
/// Main program
///
/// \mainpage CoMD: A Classical Molecular Dynamics Mini-app
///
/// CoMD is a reference implementation of typical classical molecular
/// dynamics algorithms and workloads.  It is created and maintained by
/// The Exascale Co-Design Center for Materials in Extreme Environments
/// (ExMatEx).  http://codesign.lanl.gov/projects/exmatex.  The
/// code is intended to serve as a vehicle for co-design by allowing
/// others to extend and/or reimplement it as needed to test performance of 
/// new architectures, programming models, etc.
///
/// The current version of CoMD is available from:
/// http://exmatex.github.io/CoMD
///
/// To contact the developers of CoMD send email to: exmatex-comd@llnl.gov.
///
/// This version has been ported to SYCL+MPI+oneCCL for Intel GPUs.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <sycl/sycl.hpp>

extern "C" {
#include "CoMDTypes.h"
#include "decomposition.h"
#include "linkCells.h"
#include "neighborList.h"
#include "eam.h"
#include "ljForce.h"
#include "initAtoms.h"
#include "memUtils.h"
#include "yamlOutput.h"
#include "performanceTimers.h"
#include "mycommand.h"
#include "timestep.h"
#include "constants.h"
}

#include "defines.h"
#include "parallel.h"
#include "gpu_utility.h"

#define REDIRECT_OUTPUT 0
#define   MIN(A,B) ((A) < (B) ? (A) : (B))

static SimFlat* initSimulation(Command cmd);
static void destroySimulation(SimFlat** ps);

static void initSubsystems(void);
static void finalizeSubsystems(void);

static BasePotential* initPotential(
   int doeam, const char* potDir, const char* potName, const char* potType);
static SpeciesData* initSpecies(BasePotential* pot);
static Validate* initValidate(SimFlat* s);
static void validateResult(const Validate* val, SimFlat *sim);

static void sumAtoms(SimFlat* s);
static void printThings(SimFlat* s, int iStep, double elapsedTime);
static void printSimulationDataYaml(FILE* file, SimFlat* s);
static void sanityChecks(Command cmd, double cutoff, double latticeConst, char latticeType[8]);

int main(int argc, char** argv)
{
   // Prolog
   initParallel(&argc, &argv);
   profileStart(totalTimer);
   initSubsystems();
   timestampBarrier("Starting Initialization\n");

   yamlAppInfo(yamlFile);
   yamlAppInfo(screenOut);

   Command cmd = parseCommandLine(argc, argv);
   printCmdYaml(yamlFile, &cmd);
   printCmdYaml(screenOut, &cmd);

   // select device, print info, etc.
#ifdef DO_MPI
   // Get number of SYCL devices on current node
   auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
   int numGpus = (int)devices.size();
   if (numGpus == 0) {
      // Fallback to CPU if no GPU found
      numGpus = 1;
   }

   // set active device (assuming homogenous config)
   int deviceId = getMyRank() % numGpus;
   SetupGpu(deviceId);
#else
   SetupGpu(0);
#endif

   SimFlat* sim = initSimulation(cmd);
   printSimulationDataYaml(yamlFile, sim);
   printSimulationDataYaml(screenOut, sim);

   Validate* validate = initValidate(sim); // atom counts, energy	   
   timestampBarrier("Initialization Finished\n");

   timestampBarrier("Starting simulation\n");

   // This is the CoMD main loop
   const int nSteps = sim->nSteps;
   const int printRate = sim->printRate;
   int iStep = 0;
   profileStart(loopTimer);
   for (; iStep<nSteps;)
   {
      startTimer(commReduceTimer);
      sumAtoms(sim);
      stopTimer(commReduceTimer);

      printThings(sim, iStep, getElapsedTime(timestepTimer));

      startTimer(timestepTimer);
      timestep(sim, printRate, sim->dt);
      stopTimer(timestepTimer);
#if 0
      // analyze input distribution, note this is done on CPU (slow)
      AnalyzeInput(sim, iStep);
#endif     
      iStep += printRate;
   }
   profileStop(loopTimer);

   sumAtoms(sim);
   printThings(sim, iStep, getElapsedTime(timestepTimer));
   timestampBarrier("Ending simulation\n");

   // Epilog
   validateResult(validate, sim);
   profileStop(totalTimer);

   printPerformanceResults(sim->atoms->nGlobal, sim->printRate);
   printPerformanceResultsYaml(yamlFile);

   destroySimulation(&sim);
   comdFree(validate);
   finalizeSubsystems();

   timestampBarrier("CoMD Ending\n");
   destroyParallel();

   // SYCL cleanup: the queue destructor handles device reset
   // No explicit reset needed like cudaDeviceReset()

   return 0;
}

/// Initialized the main CoMD data stucture, SimFlat, based on command
/// line input from the user.  Also performs certain sanity checks on
/// the input to screen out certain non-sensical inputs.
///
/// Simple data members such as the time step dt are initialized
/// directly, substructures such as the potential, the link cells, the
/// atoms, etc., are initialized by calling additional initialization
/// functions (initPotential(), initLinkCells(), initAtoms(), etc.).
/// Initialization order is set by the natural dependencies of the
/// substructure such as the atoms need the link cells so the link cells
/// must be initialized before the atoms.
SimFlat* initSimulation(Command cmd)
{
   SimFlat* sim = (SimFlat*)comdMalloc(sizeof(SimFlat));
   sim->nSteps = cmd.nSteps;
   sim->printRate = cmd.printRate;
   sim->dt = cmd.dt;
   sim->domain = NULL;
   sim->boxes = NULL;
   sim->atoms = NULL;
   sim->ePotential = 0.0;
   sim->eKinetic = 0.0;
   sim->atomExchange = NULL;
   sim->gpuAsync = cmd.gpuAsync;
   sim->gpuProfile = cmd.gpuProfile;
  
   // if profile mode enabled: force 0 steps and turn async off
   if (sim->gpuProfile) { 
     sim->nSteps = 0;
   }

   if (!strcmp(cmd.method, "thread_atom")) sim->method = THREAD_ATOM;
   else if (!strcmp(cmd.method, "warp_atom")) sim->method = WARP_ATOM;
   else if (!strcmp(cmd.method, "warp_atom_nl")) sim->method = WARP_ATOM_NL;
   else if (!strcmp(cmd.method, "cta_cell")) sim->method = CTA_CELL;
   else if (!strcmp(cmd.method, "thread_atom_nl")) sim->method = THREAD_ATOM_NL;
   else if (!strcmp(cmd.method, "cpu_nl")) sim->method = CPU_NL;
   else {printf("Error: You have to specify a valid method: -m [thread_atom,thread_atom_nl,warp_atom,warp_atom_nl,cta_cell,cpu_nl]\n"); exit(-1);}

   int useNL = (sim->method == THREAD_ATOM_NL || sim->method == WARP_ATOM_NL || sim->method == CPU_NL)? 1 : 0;
   sim->pot = initPotential(cmd.doeam, cmd.potDir, cmd.potName, cmd.potType);

   real_t latticeConstant = cmd.lat;
   if (cmd.lat < 0.0)
      latticeConstant = sim->pot->lat;

   // ensure input parameters make sense.
   sanityChecks(cmd, sim->pot->cutoff, latticeConstant, sim->pot->latticeType);

   sim->species = initSpecies(sim->pot);

   real3_old globalExtent;
   globalExtent[0] = cmd.nx * latticeConstant;
   globalExtent[1] = cmd.ny * latticeConstant;
   globalExtent[2] = cmd.nz * latticeConstant;

   sim->domain = initDecomposition(
      cmd.xproc, cmd.yproc, cmd.zproc, globalExtent);

   sim->usePairlist = cmd.usePairlist;
   if(sim->usePairlist)
   {
       sim->gpu.atoms.neighborList.forceRebuildFlag = 1;
   }
   sim->gpu.usePairlist = sim->usePairlist;

   real_t skinDistance;
   if(useNL || sim->usePairlist){
          skinDistance = sim->pot->cutoff * cmd.relativeSkinDistance; 
          if (printRank())
                  printf("Skin-Distance: %f\n",skinDistance);
   } else
          skinDistance = 0.0;
   sim->skinDistance = skinDistance;
   if(sim->usePairlist)
       sim->gpu.atoms.neighborList.skinDistanceHalf2 = skinDistance*skinDistance/4;
   sim->boxes = initLinkCells(sim->domain, sim->pot->cutoff + skinDistance, cmd.doHilbert);
   sim->atoms = initAtoms(sim->boxes, skinDistance);

   sim->ljInterpolation = cmd.ljInterpolation;
   sim->spline = cmd.spline;

   // create lattice with desired temperature and displacement.
   createFccLattice(cmd.nx, cmd.ny, cmd.nz, latticeConstant, sim);
   setTemperature(sim, cmd.temperature);
   randomDisplacements(sim, cmd.initialDelta);

   // set atoms exchange function
   sim->atomExchange = initAtomHaloExchange(sim->domain, sim->boxes);
    if(!cmd.doeam)
    {
        SetBoundaryCells(sim, sim->atomExchange);
    }
   // set forces exchange function
   if (cmd.doeam && sim->method < CPU_NL) {
     EamPotential* pot = (EamPotential*) sim->pot;
     pot->forceExchange = initForceHaloExchange(sim->domain, sim->boxes,sim->method < CPU_NL);
     // init boundary cell lists
     SetBoundaryCells(sim, pot->forceExchange);
   }
   if((sim->method == THREAD_ATOM_NL || sim->method == WARP_ATOM_NL) && !cmd.doeam){
           if (printRank())
                   printf("Gpu neighborlist implementation is currently only supported for the eam potential.\n");
           exit(-1);
   }
 
   // setup GPU
   AllocateGpu(sim, cmd.doeam, skinDistance); 
   CopyDataToGpu(sim, cmd.doeam);

   // Forces must be computed before we call the time stepper.
   if (!sim->gpuProfile) {
     startTimer(redistributeTimer);
     redistributeAtoms(sim);
     stopTimer(redistributeTimer); 
   }

   if(useNL){
      buildNeighborList(sim,0);
   }

   startTimer(computeForceTimer);
   computeForce(sim);
   stopTimer(computeForceTimer);

   if(sim->method < CPU_NL)
      kineticEnergyGpu(sim);
   else
      kineticEnergy(sim);

   if(sim->gpuAsync != 0 && useNL){
           printf("Async Neighborlist not supported yet!\n");
           exit(-1);
   }
   return sim;
}

/// frees all data associated with *ps and frees *ps
void destroySimulation(SimFlat** ps)
{
   if ( ! ps ) return;

   SimFlat* s = *ps;
   if ( ! s ) return;

   // free GPU data
   DestroyGpu(s);

   BasePotential* pot = s->pot;
   if ( pot) pot->destroy(&pot);
   destroyLinkCells(&(s->boxes));
   destroyAtoms(s->atoms);
   destroyHaloExchange(&(s->atomExchange));
   comdFree(s->species);
   comdFree(s->domain);
   comdFree(s);
   *ps = NULL;

   return;
}

void initSubsystems(void)
{
#if REDIRECT_OUTPUT
   freopen("testOut.txt","w",screenOut);
#endif

   yamlBegin();
}

void finalizeSubsystems(void)
{
#if REDIRECT_OUTPUT
   fclose(screenOut);
#endif
   yamlEnd();
}

/// decide whether to get LJ or EAM potentials
BasePotential* initPotential(
   int doeam, const char* potDir, const char* potName, const char* potType)
{
   BasePotential* pot = NULL;

   if (doeam) 
      pot = initEamPot(potDir, potName, potType);
   else 
      pot = initLjPot();
   assert(pot);
   return pot;
}

SpeciesData* initSpecies(BasePotential* pot)
{
   SpeciesData* species = (SpeciesData*)comdMalloc(sizeof(SpeciesData));

   strcpy(species->name, pot->name);
   species->atomicNo = pot->atomicNo;
   species->mass = pot->mass;

   return species;
}

Validate* initValidate(SimFlat* sim)
{
   sumAtoms(sim);
   Validate* val = (Validate*)comdMalloc(sizeof(Validate));
   val->eTot0 = (sim->ePotential + sim->eKinetic) / sim->atoms->nGlobal;
   val->nAtoms0 = sim->atoms->nGlobal;

   if (printRank())
   {
      fprintf(screenOut, "\n");
      printSeparator(screenOut);
      fprintf(screenOut, "Initial energy : %14.12f, atom count : %d \n", 
            val->eTot0, val->nAtoms0);
      fprintf(screenOut, "\n");
   }
   return val;
}

void validateResult(const Validate* val, SimFlat* sim)
{
   if (printRank())
   {
      real_t eFinal = (sim->ePotential + sim->eKinetic) / sim->atoms->nGlobal;

      int nAtomsDelta = (sim->atoms->nGlobal - val->nAtoms0);

      fprintf(screenOut, "\n");
      fprintf(screenOut, "\n");
      fprintf(screenOut, "Simulation Validation:\n");

      fprintf(screenOut, "  Initial energy  : %14.12f\n", val->eTot0);
      fprintf(screenOut, "  Final energy    : %14.12f\n", eFinal);
      fprintf(screenOut, "  eFinal/eInitial : %f\n", eFinal/val->eTot0);
      if ( nAtomsDelta == 0)
      {
         fprintf(screenOut, "  Final atom count : %d, no atoms lost\n",
               sim->atoms->nGlobal);
      }
      else
      {
         fprintf(screenOut, "#############################\n");
         fprintf(screenOut, "# WARNING: %6d atoms lost #\n", nAtomsDelta);
         fprintf(screenOut, "#############################\n");
      }
   }
}

void sumAtoms(SimFlat* s)
{
   // sum atoms across all processers
   s->atoms->nLocal = 0;
   for (int i = 0; i < s->boxes->nLocalBoxes; i++)
   {
      s->atoms->nLocal += s->boxes->nAtoms[i];
   }

   startTimer(commReduceTimer);
   addIntParallel(&s->atoms->nLocal, &s->atoms->nGlobal, 1);
   stopTimer(commReduceTimer);
}

/// Prints current time, energy, performance etc to monitor the state of
/// the running simulation.  Performance per atom is scaled by the
/// number of local atoms per process this should give consistent timing
/// assuming reasonable load balance
void printThings(SimFlat* s, int iStep, double elapsedTime)
{
   // keep track previous value of iStep so we can calculate number of steps.
   static int iStepPrev = -1; 
   static int firstCall = 1;

   int nEval = iStep - iStepPrev; // gives nEval = 1 for zeroth step.
   iStepPrev = iStep;
   
   if (! printRank() )
      return;

   if (firstCall)
   {
      firstCall = 0;
      fprintf(screenOut, 
       "#                                                                                         Performance\n" 
       "#  Loop   Time(fs)       Total Energy   Potential Energy     Kinetic Energy  Temperature   (us/atom)     # Atoms\n");
      fflush(screenOut);
   }

   real_t time = iStep*s->dt;
   real_t eTotal = (s->ePotential+s->eKinetic) / s->atoms->nGlobal;
   real_t eK = s->eKinetic / s->atoms->nGlobal;
   real_t eU = s->ePotential / s->atoms->nGlobal;
   real_t Temp = (s->eKinetic / s->atoms->nGlobal) / (kB_eV * 1.5);

   double timePerAtom = 1.0e6*elapsedTime/(double)(nEval*s->atoms->nLocal);

   fprintf(screenOut, " %6d %10.2f %18.12f %18.12f %18.12f %12.4f %10.4f %12d\n",
           iStep, time, eTotal, eU, eK, Temp, timePerAtom, s->atoms->nGlobal);
}

/// Print information about the simulation in a format that is (mostly)
/// YAML compliant.
void printSimulationDataYaml(FILE* file, SimFlat* s)
{
   // All ranks get maxOccupancy
   int maxOcc = maxOccupancy(s->boxes);

   // Only rank 0 prints
   if (! printRank())
      return;
   
   fprintf(file,"Simulation data: \n");
   fprintf(file,"  Total atoms        : %d\n", 
           s->atoms->nGlobal);
   fprintf(file,"  Min global bounds  : [ %14.10f, %14.10f, %14.10f ]\n",
           s->domain->globalMin[0], s->domain->globalMin[1], s->domain->globalMin[2]);
   fprintf(file,"  Max global bounds  : [ %14.10f, %14.10f, %14.10f ]\n",
           s->domain->globalMax[0], s->domain->globalMax[1], s->domain->globalMax[2]);
   printSeparator(file);
   fprintf(file,"Decomposition data: \n");
   fprintf(file,"  Processors         : %6d,%6d,%6d\n", 
           s->domain->procGrid[0], s->domain->procGrid[1], s->domain->procGrid[2]);
   fprintf(file,"  Local boxes        : %6d,%6d,%6d = %8d\n", 
           s->boxes->gridSize[0], s->boxes->gridSize[1], s->boxes->gridSize[2], 
           s->boxes->gridSize[0]*s->boxes->gridSize[1]*s->boxes->gridSize[2]);
   fprintf(file,"  Box size           : [ %14.10f, %14.10f, %14.10f ]\n", 
           s->boxes->boxSize[0], s->boxes->boxSize[1], s->boxes->boxSize[2]);
   fprintf(file,"  Box factor         : [ %14.10f, %14.10f, %14.10f ] \n", 
           s->boxes->boxSize[0]/s->pot->cutoff,
           s->boxes->boxSize[1]/s->pot->cutoff,
           s->boxes->boxSize[2]/s->pot->cutoff);
   fprintf(file, "  Max Link Cell Occupancy: %d of %d\n",
           maxOcc, MAXATOMS);
   printSeparator(file);
   fprintf(file,"Potential data: \n");
   s->pot->print(file, s->pot);
   
   // Memory footprint diagnostics
   int perAtomSize = 10*sizeof(real_t)+2*sizeof(int);
   float mbPerAtom = perAtomSize/1024/1024;
   float totalMemLocal = (float)(perAtomSize*s->atoms->nLocal)/1024/1024;
   float totalMemGlobal = (float)(perAtomSize*s->atoms->nGlobal)/1024/1024;

   int nLocalBoxes = s->boxes->gridSize[0]*s->boxes->gridSize[1]*s->boxes->gridSize[2];
   int nTotalBoxes = (s->boxes->gridSize[0]+2)*(s->boxes->gridSize[1]+2)*(s->boxes->gridSize[2]+2);
   float paddedMemLocal = (float) nLocalBoxes*(perAtomSize*MAXATOMS)/1024/1024;
   float paddedMemTotal = (float) nTotalBoxes*(perAtomSize*MAXATOMS)/1024/1024;

   printSeparator(file);
   fprintf(file,"Memory data: \n");
   fprintf(file, "  Intrinsic atom footprint = %4d B/atom \n", perAtomSize);
   fprintf(file, "  Total atom footprint     = %7.3f MB (%6.2f MB/node)\n", totalMemGlobal, totalMemLocal);
   fprintf(file, "  Link cell atom footprint = %7.3f MB/node\n", paddedMemLocal);
   fprintf(file, "  Link cell atom footprint = %7.3f MB/node (including halo cell data\n", paddedMemTotal);

   fflush(file);      
}

/// Check that the user input meets certain criteria.
void sanityChecks(Command cmd, double cutoff, double latticeConst, char latticeType[8])
{
   int failCode = 0;

   // Check that domain grid matches number of ranks. (fail code 1)
   int nProcs = cmd.xproc * cmd.yproc * cmd.zproc;
   if (nProcs != getNRanks())
   {
      failCode |= 1;
      if (printRank() )
         fprintf(screenOut,
                 "\nNumber of MPI ranks must match xproc * yproc * zproc\n");
   }

   // Check whether simuation is too small (fail code 2)
   double minx = 2*cutoff*cmd.xproc;
   double miny = 2*cutoff*cmd.yproc;
   double minz = 2*cutoff*cmd.zproc;
   double sizex = cmd.nx*latticeConst;
   double sizey = cmd.ny*latticeConst;
   double sizez = cmd.nz*latticeConst;

   if ( sizex < minx || sizey < miny || sizez < minz)
   {
      failCode |= 2;
      if (printRank())
         fprintf(screenOut,"\nSimulation too small.\n"
                 "  Increase the number of unit cells to make the simulation\n"
                 "  at least (%3.2f, %3.2f. %3.2f) Ansgstroms in size\n",
                 minx, miny, minz);
   }

   // Check for supported lattice structure (fail code 4)
   if (strcasecmp(latticeType, "FCC") != 0)
   {
      failCode |= 4;
      if ( printRank() )
         fprintf(screenOut,
                 "\nOnly FCC Lattice type supported, not %s. Fatal Error.\n",
                 latticeType);
   }
   int checkCode = failCode;
   bcastParallel(&checkCode, sizeof(int), 0);
   // This assertion can only fail if different tasks failed different
   // sanity checks.  That should not be possible.
   assert(checkCode == failCode);
      
   if (failCode != 0)
      exit(failCode);
}
