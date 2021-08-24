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
