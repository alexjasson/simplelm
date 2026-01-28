/*
 * SimpleLM
 * © 2026 Alex Jasson
 */

#include <stdio.h>
#include "Model.h"

#define MODEL "data/model.out"

int main(int argc, char **argv)
{
    Model m = ModelRead(MODEL);
    
    // Chat loop

    ModelFree(m); 
    return 0;
}