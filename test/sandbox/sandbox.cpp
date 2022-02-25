

#include "sandbox.hpp"

#include <cppitertools/enumerate.hpp>
#include <cppitertools/reversed.hpp>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <string>
#include <utility>

#include "nor/nor.hpp"

namespace nor {

struct S {
   S(std::string k) : str(std::move(k)) {}
   S(S&& s) noexcept { std::cout << " Moved! "; };
   S(const S& s) { std::cout << " Copied! "; };

   nor::games::stratego::sandbox s;
   std::string str;
};
}

int main(int argc, char** argv)
{

   S s{""};
   std::cout << std::is_move_assignable_v< S > << std::endl;
   S s2{nor::utils::move_if< std::is_move_constructible >(s)};
   std::cout << s.str;

}