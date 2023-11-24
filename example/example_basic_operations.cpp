#include <iostream>
#include <string>
#include <casadi/casadi.hpp>

namespace cs = casadi;

int main()
{
    //symbolic vector
    cs::SX q = cs::SX::sym("q", 7);
    std::cout << "q: " << q << std::endl;
    
    return 0;
}