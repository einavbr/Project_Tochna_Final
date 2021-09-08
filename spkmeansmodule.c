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

static PyObject* pythonRunDdgFlow(PyObject * self, PyObject * args){
    int i;
    char* k, *file_name;
    Graph* graph;
    PyObject *PyDiagDegreeArray;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    PyDiagDegreeArray = PyList_New(graph->size);
    if (!PyDiagDegreeArray){
        return NULL;
    }
    for (i = 0; i < graph->size; i++){
        PyList_SET_ITEM(PyDiagDegreeArray, i, Py_BuildValue("d", graph->diagonal_degree_array[i]));
    }

    freeGraph(graph);

    return PyDiagDegreeArray;
}

static PyObject* pythonRunLnormFlow(PyObject * self, PyObject * args){
    int i, j;
    char* k, *file_name;
    Graph* graph;
    double** laplacian_mat;
    PyObject *PyLaplacianMat, *PyLaplacianRow;

    if(!PyArg_ParseTuple(args, "ss", &k, &file_name)){
        return NULL;
    }
    graph = pythonGraphInit(k, file_name);
    if (graph == NULL){
        return NULL;
    }
    laplacian_mat = allocateMatrix(graph->size, graph->size);
    assert(laplacian_mat && ERROR_OCCURED);
    runLnormFlow(graph, laplacian_mat, FALSE);
    PyLaplacianMat = PyList_New(graph->size);
    if (!PyLaplacianMat){
        return NULL;
    }
    for (i = 0; i < graph->size; i++){
        PyLaplacianRow = PyList_New(graph->size);
        for (j = 0; j < graph->size; j++){
            PyList_SET_ITEM(PyLaplacianRow, j, Py_BuildValue("d", laplacian_mat[i][j]));
        }
        PyList_SET_ITEM(PyLaplacianMat, i, Py_BuildValue("O", PyLaplacianRow));
    }
    /* free laplacian matrix */
    freeGraph(graph);

    return PyLaplacianMat;
}

static PyObject* pythonRunJacobiFlow(PyObject * self, PyObject * args){
    int i, j;
    char* k, *file_name;
    Graph* graph;
    Eigen** eigensArray;
    double** laplacian_mat;
    PyObject *PyJacobiMat, *PyJacobiRow;

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
    runJacobiFlow(graph, laplacian_mat, eigensArray, FALSE);
    /* free laplacian matrix */
    PyJacobiMat = PyList_New(graph->size);
    if (!PyJacobiMat){
        return NULL;
    }
    for (i = 0; i < graph->size; i++){
        PyJacobiRow = PyList_New(graph->size);
        for (j = 0; j < graph->size; j++){
            PyList_SET_ITEM(PyJacobiRow, j, Py_BuildValue("d", eigensArray[i][j]));
        }
        PyList_SET_ITEM(PyJacobiMat, i, Py_BuildValue("O", PyJacobiRow));
    }
    /* free eigensArray */
    freeGraph(graph);

    return PyJacobiMat;
}

static PyObject* pythonRunSpkFlow(PyObject * self, PyObject * args){
    int i, j, N, DIM, K;
    char* k, *file_name;
    Graph* graph;
    double** T/* , ** centroids_mat */;
    /* int* whichClusterArray; */
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
static double ** initVectorsArray (PyObject *VectorsList, int size, int point_size){
    double **vectors;
    PyObject *curVector, *coordinate;
    int i, j;
    
    vectors = (double**) malloc(size * sizeof(double*));
    assert(vectors != NULL && ERROR_OCCURED);
    for (i = 0; i < size; i++){
        curVector = PyList_GetItem(VectorsList, i);
        vectors[i] = (double*) malloc(point_size * sizeof(double*));
        assert(vectors[i] != NULL && ERROR_OCCURED);
        for (j = 0; j < point_size; j++){
           coordinate = PyList_GetItem(curVector, j);
           vectors[i][j] = PyFloat_AsDouble(coordinate);
        }
    }

    return vectors;
}

static PyObject* pythonRunkmeanspp(PyObject *self, PyObject *args) {
    PyObject *pyPoints;
    PyObject *pyCentroids;
    PyObject *pyRes;
    PyObject *pyThisCentroid;
    int k, n, i, j, point_size;
    double **points, **centroids, **res;
    PyObject* repr, *str;
    char* bytes;

    if(!PyArg_ParseTuple(args, "OOiii", &pyPoints, &pyCentroids, &k, &n, &point_size)){
        return NULL;
    }
    if(pyCentroids == NULL){
        printf("baaaaa");
    }
    if (k >= n){
        printf("%s", INVALID_INPUT);
        return NULL;
    }

    printf("starting kmeans\n");
    points = initVectorsArray(pyPoints, n, point_size);
    printf("initialized points\n");
    centroids = initVectorsArray(pyCentroids, k, point_size); /* ABAAYA */
    printf("initialized centroids\n");
    res = kmeanspp(points, centroids, k, n, point_size);
    printf("finished kmeanspp\n");
    
    if (res == NULL){
        return NULL;
    }
    pyRes = PyList_New(k);
    if (!pyRes){
        return NULL;
    }
    for (i = 0; i < k; i++) {
        pyThisCentroid = PyList_New(point_size);
        for (j = 0; j < point_size; j++){
            PyList_SET_ITEM(pyThisCentroid, j, Py_BuildValue("d", res[i][j]));
        }
        PyList_SET_ITEM(pyRes, i, Py_BuildValue("O", pyThisCentroid));
    }
    printf("initialized pyRes\n");

    free_double_pointerpp(centroids, k);
    free_double_pointerpp(points, n);

    /* repr = PyObject_Repr(pyRes);
    str = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
    bytes = PyBytes_AS_STRING(str);

    printf("REPR: %s\n", bytes); */

    return PyLong_FromDouble(42);
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
