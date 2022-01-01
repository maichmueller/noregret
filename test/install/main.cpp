#include <iostream>
#include <per/per.hpp>

int main()
{
   auto per = per::PrioritizedExperience< int >{10};
   std::cout << "Install successful!" << std::endl;
   return 0;
}
