#include "spkmeans.h"
#include <Python.h> 

#define INVALID_INPUT "Invalid Input!"
#define ERROR_OCCURED "An Error Has Occured"

/** TODO:  
 * PyObject - for each function\s we'll be calling
 * PyMethodDef - 
 * PyModuleDef
 * PyInint
 * 
 * Python flow: 
 * recieves Parameters -> passes parameters to spkmeans.c through this module -> 
 * spkmeans.c does calculations and returns to python before step 6 -> Pyton calls kmeans++ (hw2) also through this module ->
 * return to spkmeans.c for step 7 and finish. 
 */
static PyObject* pythonRunWamFlow(PyObject * self, PyObject * args){
    int i, j;
    char* k, *file_name;
    Graph* graph;
    PyObject *PyWeightsMat, *PyWeightsRow;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    PyWeightsMat = PyList_New(graph->size);
    if (!PyWeightsMat){
        return NULL;
    }
    for (i = 0; i < graph->size; i++){
        PyWeightsRow = PyList_New(graph->size);
        for (j = 0; j < graph->size; j++){
            PyList_SET_ITEM(PyWeightsRow, j, Py_BuildValue("d", graph->weighted_mat[i][j]));
        }
        PyList_SET_ITEM(PyWeightsMat, i, Py_BuildValue("O", PyWeightsRow));
    }

    freeGraph(graph);

    return PyWeightsMat;
}
/*
static PyObject* pythonRunWamFlow_wrapper(PyObject * self, PyObject * args){
    char* file_name, *k;
    PyObject * ret;

    //parse arguments
    if (!PyArg_ParseTuple(args, "ss", &k, &file_name)) {
    return NULL;
  }
  printf("Hello Einav, inside module\n");
  // run the actual function
  ret = pythonRunWamFlow(k, file_name);

  return ret;
}
*/
// Method definition object for this extension
static PyMethodDef spkmeansMethods[] = { 
    {  
        "pythonRunWamFlow",
        (PyCFunction) pythonRunWamFlow,
        METH_VARARGS,
        PyDoc_STR("Run WAM flow")
    },
    {NULL, NULL, 0, NULL}
};

// Module definition
// The arguments of this structure tell Python what to call your extension,
// what it's methods are and where to look for it's method definitions
static struct PyModuleDef spkmeans_def = { 
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule", // Name for the new module
    "A Python module that currently does nothing",
    -1, 
    spkmeansMethods
};

// Module initialization
// Python calls this function when importing your extension. It is important
// that this function is named PyInit_[[your_module_name]] exactly, and matches
// the name keyword argument in setup.py's setup() call.
PyMODINIT_FUNC PyInit_spkmeansmodule(void) {
    PyObject *module;
    module = PyModule_Create(&spkmeans_def);
    if (!module){
        return NULL;
    }
    return module;
}
