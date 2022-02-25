

#include "sandbox.hpp"

#include <cppitertools/enumerate.hpp>
#include <cppitertools/reversed.hpp>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <string>

int main(int argc, char** argv)
{
   std::vector< std::string > actions{"first", "second", "third", "fourth", "fifth"};
   auto size = actions.size();
   for(auto&& [action_idx, action] : iter::enumerate(iter::reversed(actions)))
      {
         action_idx = size - 1 - action_idx;
         std::cout << "idx " << std::quoted(std::to_string(action_idx)) << ", "
                   << "action " << action << "\n";
      }
}