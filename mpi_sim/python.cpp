#include <iostream>
#include <vector>
#include <list>

#include "python.hpp"

bool is_vector(bpyn::array a){
    int ndim = bpy::extract<int>(a.attr("ndim"));
    return ndim == 1;
}

Vector* ndarray_to_vector(bpyn::array a){

#ifdef _DEBUG
    std::cout << "Extracting vector:" << std::endl;
#endif

    int size = bpy::extract<int>(a.attr("size"));
    Vector* ret = new Vector(size);
    for(unsigned i = 0; i < size; i++){
        (*ret)(i) = bpy::extract<float>(a[i]);
    }

#ifdef _DEBUG
    std::cout << "Size:" << size << std::endl;
    std::cout << "Value:" << std::endl;
    std::cout << *ret << std::endl << std::endl;
#endif

    return ret;
}

Matrix* ndarray_to_matrix(bpyn::array a){

#ifdef _DEBUG
    std::cout << "Extracting matrix:" << std::endl;
#endif

    int ndim = bpy::extract<int>(a.attr("ndim"));
    int size = bpy::extract<int>(a.attr("size"));
    std::vector<int> shape(ndim);
    std::vector<int> strides(ndim);
    bpy::object python_shape = a.attr("shape");
    bpy::object python_strides = a.attr("strides");
    for(unsigned i = 0; i < ndim; i++){ 
        shape[i] = bpy::extract<int>(python_shape[i]);
        strides[i] = bpy::extract<int>(python_strides[i]);
    }

    Matrix* ret = new Matrix(shape[0], shape[1]);
    for(unsigned i = 0; i < shape[0]; i++){
        for(unsigned j = 0; j < shape[1]; j++){
            (*ret)(i, j) = bpy::extract<float>(a[i][j]);
        }
    }

#ifdef _DEBUG
    std::cout << "Ndim:" << ndim << std::endl;
    std::cout << "Size:" << size << std::endl;
    std::cout << "Shape:" << std::endl;
    std::cout << "(";
    for(unsigned i = 0; i < ndim; i++){
        std::cout << shape[i] << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "Value:" << std::endl;
    std::cout << *ret << std::endl << std::endl;
#endif

    return ret;
}


PythonMpiSimulatorChunk::PythonMpiSimulatorChunk(){
}

PythonMpiSimulatorChunk::PythonMpiSimulatorChunk(double dt)
    :mpi_sim_chunk(dt){
}

void PythonMpiSimulatorChunk::run_n_steps(bpy::object pysteps){
    int steps = bpy::extract<int>(pysteps);
    mpi_sim_chunk.run_n_steps(steps);
}

void PythonMpiSimulatorChunk::add_signal(bpy::object key, bpyn::array sig){
    if( is_vector(sig) ){
        Vector* vec = ndarray_to_vector(sig);
        mpi_sim_chunk.add_vector_signal(bpy::extract<key_type>(key), vec);
    }else{
        Matrix* mat = ndarray_to_matrix(sig);
        mpi_sim_chunk.add_matrix_signal(bpy::extract<key_type>(key), mat);
    }
}

bpy::object PythonMpiSimulatorChunk::get_probe_data(bpy::object probe_key, bpy::object make_array){
    key_type c_probe_key = bpy::extract<key_type>(probe_key);
    Probe<Vector>* probe = mpi_sim_chunk.get_probe(c_probe_key);
    list<Vector*> data = probe->get_data();

    bpy::list py_list;
    list<Vector*>::const_iterator it; 
    for(it = data.begin(); it != data.end(); ++it){

        //bpyn::array a(**it);
        //bpy::object a((*it)->size());
        bpy::object a = make_array((*it)->size());
        for(unsigned i=0; i < (*it)->size(); ++i){
            a[i] = (**it)[i];
        }

        py_list.append(a);
    }

    return py_list;
}

void PythonMpiSimulatorChunk::create_Probe(bpy::object key, bpy::object signal, bpy::object period){
    key_type signal_key = bpy::extract<key_type>(signal);
    Vector* signal_vec = mpi_sim_chunk.get_vector_signal(signal_key);
    int c_period = bpy::extract<int>(period);

    Probe<Vector>* probe = new Probe<Vector>(signal_vec, c_period);

    key_type c_key = bpy::extract<key_type>(key);
    mpi_sim_chunk.add_probe(c_key, probe);
}

void PythonMpiSimulatorChunk::create_Reset(bpy::object dst, bpy::object value){
    key_type dst_key = bpy::extract<key_type>(dst);
    float c_value = bpy::extract<float>(value);

    Vector* dst_vec = mpi_sim_chunk.get_vector_signal(dst_key);

    Operator* reset = new Reset(dst_vec, c_value);
    mpi_sim_chunk.add_operator(reset);
}

void PythonMpiSimulatorChunk::create_Copy(bpy::object dst, bpy::object src){
    key_type dst_key = bpy::extract<key_type>(dst);
    key_type src_key = bpy::extract<key_type>(src);

    Vector* dst_vec = mpi_sim_chunk.get_vector_signal(dst_key);
    Vector* src_vec = mpi_sim_chunk.get_vector_signal(src_key);

    Operator* copy = new Copy(dst_vec, src_vec);
    mpi_sim_chunk.add_operator(copy);
}

void PythonMpiSimulatorChunk::create_DotInc(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk.get_matrix_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* dot_inc = new DotInc(A_mat, X_vec, Y_vec);
    mpi_sim_chunk.add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_ScalarDotInc(bpy::object A, bpy::object X, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type Y_key = bpy::extract<key_type>(Y);

    Vector* A_scalar = mpi_sim_chunk.get_vector_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* dot_inc = new ScalarDotInc(A_scalar, X_vec, Y_vec);
    mpi_sim_chunk.add_operator(dot_inc);
}

void PythonMpiSimulatorChunk::create_ProdUpdate(bpy::object A, bpy::object X, bpy::object B, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type B_key = bpy::extract<key_type>(B);
    key_type Y_key = bpy::extract<key_type>(Y);

    Matrix* A_mat = mpi_sim_chunk.get_matrix_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* B_vec = mpi_sim_chunk.get_vector_signal(B_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* prod_update = new ProdUpdate(A_mat, X_vec, B_vec, Y_vec);
    mpi_sim_chunk.add_operator(prod_update);
}

void PythonMpiSimulatorChunk::create_ScalarProdUpdate(bpy::object A, bpy::object X, bpy::object B, bpy::object Y){
    key_type A_key = bpy::extract<key_type>(A);
    key_type X_key = bpy::extract<key_type>(X);
    key_type B_key = bpy::extract<key_type>(B);
    key_type Y_key = bpy::extract<key_type>(Y);

    Vector* A_scalar = mpi_sim_chunk.get_vector_signal(A_key);
    Vector* X_vec = mpi_sim_chunk.get_vector_signal(X_key);
    Vector* B_vec = mpi_sim_chunk.get_vector_signal(B_key);
    Vector* Y_vec = mpi_sim_chunk.get_vector_signal(Y_key);

    Operator* scalar_prod_update = new ScalarProdUpdate(A_scalar, X_vec, B_vec, Y_vec);
    mpi_sim_chunk.add_operator(scalar_prod_update);
}

void PythonMpiSimulatorChunk::create_SimLIF(bpy::object n_neurons, bpy::object tau_rc, 
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    float c_tau_rc = bpy::extract<float>(tau_rc);
    float c_tau_ref = bpy::extract<float>(tau_ref);
    float c_dt = bpy::extract<float>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Vector* J_vec = mpi_sim_chunk.get_vector_signal(J_key);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Operator* sim_lif = new SimLIF(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_vec, output_vec);
    mpi_sim_chunk.add_operator(sim_lif);
}

void PythonMpiSimulatorChunk::create_SimLIFRate(bpy::object n_neurons, bpy::object tau_rc, 
    bpy::object tau_ref, bpy::object dt, bpy::object J, bpy::object output){

    int c_n_neurons = bpy::extract<int>(n_neurons);
    float c_tau_rc = bpy::extract<float>(tau_rc);
    float c_tau_ref = bpy::extract<float>(tau_ref);
    float c_dt = bpy::extract<float>(dt);

    key_type J_key = bpy::extract<key_type>(J);
    key_type output_key = bpy::extract<key_type>(output);

    Vector* J_vec = mpi_sim_chunk.get_vector_signal(J_key);
    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    Operator* sim_lif_rate = new SimLIFRate(c_n_neurons, c_tau_rc, c_tau_ref, c_dt, J_vec, output_vec);
    mpi_sim_chunk.add_operator(sim_lif_rate);
}

void PythonMpiSimulatorChunk::create_MPISend(){}

void PythonMpiSimulatorChunk::create_MPIReceive(){}

void PythonMpiSimulatorChunk::create_PyFunc(bpy::object output, bpy::object py_fn, bpy::object t_in){
    bool c_t_in = bpy::extract<bool>(t_in);

    key_type output_key = bpy::extract<key_type>(output);

    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);

    double* time_pointer = c_t_in ? mpi_sim_chunk.get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(output_vec, py_fn, time_pointer);
    mpi_sim_chunk.add_operator(sim_py_func);
}

void PythonMpiSimulatorChunk::create_PyFuncWithInput(bpy::object output, bpy::object py_fn, 
    bpy::object t_in, bpy::object input, bpyn::array py_input){

    bool c_t_in = bpy::extract<bool>(t_in);

    key_type output_key = bpy::extract<key_type>(output);
    key_type input_key = bpy::extract<key_type>(input);

    Vector* output_vec = mpi_sim_chunk.get_vector_signal(output_key);
    Vector* input_vec = mpi_sim_chunk.get_vector_signal(input_key);

    double* time_pointer = c_t_in ? mpi_sim_chunk.get_time_pointer() : NULL;

    Operator* sim_py_func = new PyFunc(output_vec, py_fn, time_pointer, input_vec, py_input);
    mpi_sim_chunk.add_operator(sim_py_func);
}



PyFunc::PyFunc(Vector* output, bpy::object py_fn, double* time)
    :output(output), py_fn(py_fn), time(time), supply_time(time!=NULL), supply_input(false), 
	input(NULL), py_input(0.0){
}

PyFunc::PyFunc(Vector* output, bpy::object py_fn, double* time, Vector* input, bpyn::array py_input)
    :output(output), py_fn(py_fn),  time(time), supply_time(time!=NULL), supply_input(true), 
    input(input), py_input(py_input){
}

void PyFunc::operator() (){
    
    bpy::object py_output;
    if(supply_input){
        for(unsigned i = 0; i < input->size(); ++i){
            py_input[i] = (*input)[i];
        }

        if(supply_time){
            py_output = py_fn(*time, py_input);
        }else{
            py_output = py_fn(py_input);
        }
    }else{
        if(supply_time){
            py_output = py_fn(*time);
        }else{
            py_output = py_fn();
        }
    }

    try {
        (*output)[0] = bpy::extract<float>(py_output);
    } catch (const bpy::error_already_set& e) {
        for(unsigned i = 0; i < output->size(); ++i){
            (*output)[i] = bpy::extract<float>(py_output[i]);
        }
    }

#ifdef _DEBUG
    cout << *this;
#endif
}

void PyFunc::print(ostream &out) const{
    out << "PyFunc: " << endl;
    out << "Output: " << endl;
    out << *output << endl << endl;
}

BOOST_PYTHON_MODULE(mpi_sim)
{
    bpy::numeric::array::set_module_and_type("numpy", "ndarray");
    bpy::class_<PythonMpiSimulatorChunk>("PythonMpiSimulatorChunk", bpy::init<>())
        .def(bpy::init<double>())
        .def("run_n_steps", &PythonMpiSimulatorChunk::run_n_steps)
        .def("add_signal", &PythonMpiSimulatorChunk::add_signal)
        .def("get_probe_data", &PythonMpiSimulatorChunk::get_probe_data)
        .def("create_Probe", &PythonMpiSimulatorChunk::create_Probe)
        .def("create_Reset", &PythonMpiSimulatorChunk::create_Reset)
        .def("create_Copy", &PythonMpiSimulatorChunk::create_Copy)
        .def("create_DotInc", &PythonMpiSimulatorChunk::create_DotInc)
        .def("create_ScalarDotInc", &PythonMpiSimulatorChunk::create_ScalarDotInc)
        .def("create_ProdUpdate", &PythonMpiSimulatorChunk::create_ProdUpdate)
        .def("create_ScalarProdUpdate", &PythonMpiSimulatorChunk::create_ScalarProdUpdate)
        .def("create_SimLIF", &PythonMpiSimulatorChunk::create_SimLIF)
        .def("create_SimLIFRate", &PythonMpiSimulatorChunk::create_SimLIFRate)
        .def("create_MPISend", &PythonMpiSimulatorChunk::create_MPISend)
        .def("create_MPIReceive", &PythonMpiSimulatorChunk::create_MPIReceive)
        .def("create_PyFunc", &PythonMpiSimulatorChunk::create_PyFunc)
        .def("create_PyFuncWithInput", &PythonMpiSimulatorChunk::create_PyFuncWithInput);
}
