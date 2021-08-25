#include "spkmeans.h"
#include <Python.h>

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

static PyObject* printTest_wrapper(PyObject * self, PyObject * args){
    int input;
    int result;
    PyObject * ret;

    //parse arguments
    if (!PyArg_ParseTuple(args, "i", &input)) {
    return NULL;
  }
  printf("Hello Keren, inside module\n");
  // run the actual function
  result = printTest(input);
  // build the resulting string into a Python object.
  ret = PyLong_FromLong(result);

  return ret;
}

// Method definition object for this extension
static PyMethodDef spkmeansMethods[] = { 
    {  
        "printTest",
        (PyCFunction) printTest_wrapper,
        METH_VARARGS,
        PyDoc_STR("Testing module with simple function")
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