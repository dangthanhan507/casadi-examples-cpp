#include <iostream>
#include <string>
#include <casadi/casadi.hpp>
#include <casadi/core/sparsity_interface.hpp>

namespace cs = casadi;

int main()
{
    //defining symbolic vector
    cs::SX q = cs::SX::sym("q", 7);
    std::cout << "q: " << q << std::endl;

    cs::SX q2 = cs::SX::sym("q2", 7);
    std::cout << "q2: " << q2 << std::endl;

    //hstack
    cs::SX qh = cs::SX::horzcat({q, q2});
    std::cout << "horizontally stack q and q2:" << qh << std::endl;

    //vstack
    cs::SX pv = cs::SX::vertcat({q, q2});
    std::cout << "vertically stack q and q2:" << pv << std::endl;
    

    //transpose
    // cs::SX qhT = qh.T();
    cs::SX qhT = transpose(qh);
    std::cout << "transpose q:" << qhT << std::endl;

    //symbolic vectors and change values
    cs::SX t1 = cs::SX::sym("t1", 3);
    t1(0) = 1;
    t1(1) = 2;
    t1(2) = 0.1564;

    cs::SX t2 = cs::SX::zeros(3, 1);
    t2(1) = 0.1564;
    t2(2) = -2;
    std::cout << "t1: " << t1 << std::endl;
    std::cout << "t2: " << t2 << std::endl;

    //create symbolic matrix
    cs::SX M = cs::SX::sym("M", 4, 4);

    //slice (the numpy favorite)
    M(cs::Slice(0,3,1),3) = t1; // M[0:3, 3] = t1

    M(3,3) = 1;
    M(2,2) = -1;
    M(0,0) = cos(q(0));
    M(0,1) = sin(q(0));
    M(1,0) = -sin(q(0));
    M(1,1) = cos(q(0));
    std::cout << "M: " << M << std::endl;

    //slice
    cs::SX M2 = cs::SX::zeros(4, 4);
    M2(cs::Slice(0,3,1),3) = t2; // M[0:3, 3] = t2
    M2(3,3) = 1;
    M2(2,2) = -1;
    M2(0,0) = cos(q(1));
    M2(0,1) = sin(q(1));
    M2(1,0) = -sin(q(1));
    M2(1,1) = cos(q(1));
    std::cout << "M2: " << M2 << std::endl;

    // test matrix multiplication
    casadi::SX _A = casadi::SX::zeros(2, 2);
    _A(0,0) = 100;
    _A(0,1) = 50;
    _A(1,0) = -1;
    _A(1,1) = -2;
    casadi::SX _B = casadi::SX::zeros(2, 2);
    _B(0,0) = 0.5;
    _B(0,1) = 0.1;
    _B(1,0) = 0.2;
    _B(1,1) = 0.4;
    std::cout << "test matrix multiplication: " << std::endl;
    std::cout << "_A: " << _A << std::endl;
    std::cout << "_B: " << _B << std::endl;
    std::cout << "element-wise multiplication: " << _A * _B << std::endl;
    std::cout << "matrix multiplication: " << mtimes(_A, _B) << std::endl;

    casadi::SX T1 = M;
    casadi::SX T2 = mtimes(M, M2);
    std::cout << "T1: " << T1 << std::endl;
    std::cout << "T2: " << T2 << std::endl;
    return 0;
}