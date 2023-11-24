#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <casadi/casadi.hpp>


namespace fs = std::filesystem;
namespace cs = casadi;

int main()
{
        /*  Test problem
     *
     *    min x0^2 + x1^2
     *    s.t.    x0 + x1 - 10 = 0
     */

    //file names + prefixes
    std::string file_name = "nlp_code";
    std::string prefix_code = fs::current_path().string() + "/";
    std::string prefix_lib = fs::current_path().string() + "/";

    //create nlp solver from compiled code
    std::string lib_name = prefix_lib + file_name + ".so";
    cs::Function solver = cs::nlpsol("solver","ipopt",lib_name);

    //bounds and initial guess
    std::map<std::string, cs::DM> arg, res;
    arg["lbx"] = -cs::DM::inf();
    arg["ubx"] = cs::DM::inf();
    arg["lbg"] = 0;
    arg["ubg"] = cs::DM::inf();
    arg["x0"] = 0;

    //solve nlp
    res = solver(arg);

    //print sol
    std::cout << "-----" << std::endl;
    std::cout << "objective at solution = " << res.at("f") << std::endl;
    std::cout << "primal solution = " << res.at("x") << std::endl;
    std::cout << "dual solution (x) = " << res.at("lam_x") << std::endl;
    std::cout << "dual solution (g) = " << res.at("lam_g") << std::endl;
    return 0;
}