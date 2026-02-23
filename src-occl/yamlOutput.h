/// \file
/// Write simulation information in YAML format.

#ifndef __YAML_OUTPUT_H
#define __YAML_OUTPUT_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Provide access to the YAML file in other compilation units.
extern FILE* yamlFile;

void yamlBegin(void);
void yamlEnd(void);

void yamlAppInfo(FILE* file);

void printSeparator(FILE* file);

#ifdef __cplusplus
}
#endif

#endif
