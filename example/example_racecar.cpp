#include <iostream>
#include <casadi/casadi.hpp>

casadi::MX f(const casadi::MX& x, const casadi::MX& u)
{
    return vertcat(x(1), u-x(1));
}

namespace cs = casadi;
int main()
{
    /**
     * 1D racecar problem
     * state x: R^2
     * control u: R
     * dynamics: xdot = f(x,u) = [x_2, u-x_2]
     * 
     * run mpc problem with rk-4 integration
    */
    int N =  100;
    cs::Opti opti = cs::Opti();
    
    cs::Slice all;

    cs::MX x = opti.variable(2, N+1);
    cs::MX pos = x(0,all);
    cs::MX vel = x(1,all);
    
    cs::MX u = opti.variable(1,N);
    cs::MX T = opti.variable(); //final time

    //objective function
    opti.minimize(T); 

    cs::MX dt = T/N;
    for (int k=0;k<N;++k)
    {
        cs::MX k1 = f(x(all,k),u(k));
        cs::MX k2 = f(x(all,k)+dt/2*k1,u(k));
        cs::MX k3 = f(x(all,k)+dt/2*k2,u(k));
        cs::MX k4 = f(x(all,k)+dt*k3,u(k));
        cs::MX x_next = x(all,k) + dt/6*(k1+2*k2+2*k3+k4); //rk-4 integration
        opti.subject_to(x(all,k+1)==x_next); // run as constraint
    }


    //path constraints
    opti.subject_to(vel<=1-sin(2*cs::pi*pos)/2); //track speed limit
    opti.subject_to(0<=u<=1); //control is limited

    //boundary conditions
    opti.subject_to(pos(0)==0); //start at 0
    opti.subject_to(vel(0)==0); //start at 0
    opti.subject_to(pos(N)==1); //finish at 1

    // misc constarints
    opti.subject_to(T>=0); //time must be positive

    //initial guess
    opti.set_initial(vel,1);
    opti.set_initial(T,1);

    // solve NLP
    opti.solver("ipopt"); //use ipopt
    cs::OptiSol sol = opti.solve();
    return 0;
}