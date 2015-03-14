#include "mpi_operator.hpp"

MPISend::MPISend(int dst, int tag, SignalView content):
        dst(dst), tag(tag), content(content), first_call(true){

    content_data = &(content.data().expression().data()[0]);
    size = content.size1() * content.size2();
    buffer = new dtype[size];
}

MPIRecv::MPIRecv(int src, int tag, SignalView content):
        src(src), tag(tag), content(content), first_call(true){

    content_data = &(content.data().expression().data()[0]);
    size = content.size1() * content.size2();
    buffer = new dtype[size];
}

void MPISend::operator() (){

    if(first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);
    }

    memcpy(buffer, content_data, size * sizeof(dtype));

    MPI_Isend(buffer, size, MPI_DOUBLE, dst, tag, comm, &request);

    run_dbg(*this);
}

void MPIRecv::operator() (){

    if(first_call){
        first_call = false;
    }else{
        MPI_Wait(&request, &status);

        memcpy(content_data, buffer, size * sizeof(dtype));
    }

    MPI_Irecv(buffer, size, MPI_DOUBLE, src, tag, comm, &request);

    run_dbg(*this);
}

void MPISend::set_communicator(MPI_Comm comm){
    this->comm = comm;
}

void MPIRecv::set_communicator(MPI_Comm comm){
    this->comm = comm;
}

void MPISend::complete(){
    MPI_Wait(&request, &status);
}

void MPIRecv::complete(){
    MPI_Wait(&request, &status);
}

void MPIBarrier::operator() (){
    if(step != 0 && step % BARRIER_PERIOD == 0){
        MPI_Barrier(comm);
    }

    step++;

    run_dbg(*this);
}

string MPISend::to_string() const{
    stringstream out;

    out << "MPISend:" << endl;
    out << "tag: " << tag << endl;
    out << "dst: " << dst << endl;
    out << "size: " << size << endl;
    out << "content:" << endl;
    out << content << endl;

    out << "buffer:" << endl;

    for(int i = 0; i < size; i++){
        out << buffer[i] << ", " << endl;
    }

    return out.str();
}

string MPIRecv::to_string() const{
    stringstream out;

    out << "MPIRecv:" << endl;
    out << "tag: " << tag << endl;
    out << "src: " << src << endl;
    out << "size: " << size << endl;
    out << "content:" << endl;
    out << content << endl;

    out << "buffer:" << endl;

    for(int i = 0; i < size; i++){
        out << buffer[i] << ", " << endl;
    }

    return out.str();
}

string MPIBarrier::to_string() const{
    stringstream out;

    out << "MPIBarrier:" << endl;
    out << "step: " << step << endl;
    out << "barrier period: " << BARRIER_PERIOD << endl;

    return out.str();
}
