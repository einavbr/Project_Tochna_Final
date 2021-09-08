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

static PyObject* pythonRunWamFlow(PyObject * self, PyObject * args){
    char* k, *file_name;
    Graph* graph;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    printMatrix(graph->size, graph->size, graph->weighted_mat);

    freeGraph(graph);

    return PyLong_FromLong(42);
}

static PyObject* pythonRunDdgFlow(PyObject * self, PyObject * args){
    int i, n;
    char* k, *file_name;
    Graph* graph;
    double** ddg_mat;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    n = graph->size;
    ddg_mat = allocateMatrix(n, n);
    for (i=0 ; i<n ; i++){
        ddg_mat[i][i] = graph->diagonal_degree_array[i];
    }
    printMatrix(n, n, ddg_mat);

    freeGraph(graph);

    return PyLong_FromLong(42);
}

static PyObject* pythonRunLnormFlow(PyObject * self, PyObject * args){
    char* k, *file_name;
    Graph* graph;
    double** laplacian_mat;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    laplacian_mat = allocateMatrix(graph->size, graph->size);
    assert(laplacian_mat && ERROR_OCCURED);
    runLnormFlow(graph, laplacian_mat, TRUE);

    freeMatrix(laplacian_mat);
    freeGraph(graph);

    return PyLong_FromLong(42);
}

static PyObject* pythonRunJacobiFlow(PyObject * self, PyObject * args){
    char* k, *file_name;
    Graph* graph;
    Eigen** eigensArray;
    double** laplacian_mat;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    laplacian_mat = allocateMatrix(graph->size, graph->size);
    assert(laplacian_mat && ERROR_OCCURED);
    eigensArray = (Eigen**)malloc(graph->size * sizeof(Eigen*));
    assert(eigensArray && ERROR_OCCURED);
    runJacobiFlow(graph, laplacian_mat, eigensArray, TRUE);

    freeEigensArray(eigensArray);
    freeMatrix(laplacian_mat);
    freeGraph(graph);

    return PyLong_FromLong(42);
}

static PyObject* pythonRunSpkFlow(PyObject * self, PyObject * args){
    int i, j, N, DIM, K;
    char* k, *file_name;
    Graph* graph;
    double** T;
    PyObject *PyTMat, *PyTRow, *PyRetTuple;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    N = graph->size;
    DIM = graph->dim;
    runSpkFlowPython(graph, &K, &T);
    PyTMat = PyList_New(graph->size);
    if (!PyTMat){
        return NULL;
    }
    for (i = 0; i < graph->size; i++){
        PyTRow = PyList_New(K);
        for (j = 0; j < K; j++){
            PyList_SET_ITEM(PyTRow, j, Py_BuildValue("d", T[i][j]));
        }
        PyList_SET_ITEM(PyTMat, i, Py_BuildValue("O", PyTRow));
    }
    freeMatrix(T);
    freeGraph(graph);
    PyRetTuple = PyTuple_New(4);
    if (!PyRetTuple)
        return NULL;
    PyTuple_SET_ITEM(PyRetTuple, 0, Py_BuildValue("i", K));
    PyTuple_SET_ITEM(PyRetTuple, 1, Py_BuildValue("i", DIM));
    PyTuple_SET_ITEM(PyRetTuple, 2, Py_BuildValue("i", N));
    PyTuple_SET_ITEM(PyRetTuple, 3, PyTMat);
    if (PyErr_Occurred()) {
            Py_DECREF(PyRetTuple);
            return NULL;
        }
    return PyRetTuple;
}

// Kmeans++ Module method definitions
static double ** initVectorsArray (PyObject *VectorsList, int rows, int cols){
    double **vectors;
    PyObject *curVector, *coordinate;
    int i, j;
    
    vectors = (double**) malloc(rows * sizeof(double*));
    assert(vectors != NULL && ERROR_OCCURED);

    /* copy pyList into vectors */
    for (i = 0; i < rows; i++){
        curVector = PyList_GetItem(VectorsList, i);
        vectors[i] = (double*) malloc(cols * sizeof(double*));
        assert(vectors[i] != NULL && ERROR_OCCURED);
        for (j = 0; j < cols; j++){
            coordinate = PyList_GetItem(curVector, j);
            vectors[i][j] = PyFloat_AsDouble(coordinate);
        }
    }

    return vectors;
}

static PyObject* pythonRunkmeanspp(PyObject *self, PyObject *args) {
    PyObject *pyPoints;
    PyObject *pyCentroids;
    int k, n, point_size;
    double **points, **centroids, **res;

    if(!PyArg_ParseTuple(args, "OOiii", &pyPoints, &pyCentroids, &k, &n, &point_size)){
        return NULL;
    }
    if(pyCentroids == NULL){
        printf("%s", ERROR_OCCURED);
        return NULL;
    }
    if (k >= n){
        printf("%s", INVALID_INPUT);
        return NULL;
    }

    points = initVectorsArray(pyPoints, n, k);
    centroids = initVectorsArray(pyCentroids, k, k);
    res = kmeanspp(points, centroids, k, n, k);
    
    if (res == NULL){
        return NULL;
    }

    printMatrix(k, k, res);

    free_double_pointerpp(centroids, k);
    free_double_pointerpp(points, n);

    return PyLong_FromLong(42);
}


// Method definition object for this extension
static PyMethodDef spkmeansMethods[] = { 
    {  
        "pythonRunWamFlow",
        (PyCFunction) pythonRunWamFlow,
        METH_VARARGS,
        PyDoc_STR("Run WAM flow")
    },
    {
        "pythonRunDdgFlow",
        (PyCFunction) pythonRunDdgFlow,
        METH_VARARGS,
        PyDoc_STR("Run DDG flow")
    },
    {
        "pythonRunLnormFlow",
        (PyCFunction) pythonRunLnormFlow,
        METH_VARARGS,
        PyDoc_STR("Run Lnorm flow")
    },
    {
        "pythonRunJacobiFlow",
        (PyCFunction) pythonRunJacobiFlow,
        METH_VARARGS,
        PyDoc_STR("Run Jacobi flow")
    },
    {
        "pythonRunSpkFlow",
        (PyCFunction) pythonRunSpkFlow,
        METH_VARARGS,
        PyDoc_STR("Run SPK flow")
    },
    {  
        "pythonRunkmeanspp",
        (PyCFunction) pythonRunkmeanspp,
        METH_VARARGS,
        PyDoc_STR("Run kmeans++ flow")
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
