#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "aze/utils/utils.h"

namespace aze::arena {

struct StatTrack {
   int wins{0};
   int draws{0};
   int losses{0};
   std::vector< int > turn_counts{};

   inline void add_win(int turns = 0)
   {
      wins += 1;
      turn_counts.emplace_back(turns);
   }
   inline void add_draw(int turns = 0)
   {
      draws += 1;
      turn_counts.emplace_back(turns);
   }
};

inline void print_round_results(
   int round,
   int num_rounds,
   const std::vector< std::string > &agent_names,
   const std::vector< StatTrack > &stats)
{
   static const std::vector< std::string > colors{
      "\033[1;31m",  // RED
      "\033[1;34m",  // BLUE
      "\033[1;32m",  // GREEN
      "\033[1;33m",  // YELLOW
      "\033[1;35m",  // MAGENTA
      "\033[1;36m",  // CYAN
      "\033[1;37m"};  // WHITE
   static const std::string reset("\033[0m");

   std::stringstream ss;
   ss << "Game " << round << "/" << num_rounds;
   std::string round_display = ss.str();
   ss.clear();

   ss << "[ ";
   for(unsigned int i = 0; i < agent_names.size(); ++i) {
      const auto &color = colors[i];
      ss << "Agent " << i << " (" << color << agent_names[i] << reset << "): W: " << color
         << stats[i].wins << reset << ", ";
   }
   ss << " ]";
}

template < typename T >
inline const char *get_typename(T &object)
{
   return typeid(object).name();
}

template < typename GameType >
std::tuple< StatTrack, StatTrack > inline pit(
   GameType &game,
   int num_sims,
   bool show_game,
   int print_every_n_sim)
{
   StatTrack stats0;
   StatTrack stats1;

   for(int sim = 1; sim < num_sims; ++sim) {
      game.reset();
      LOGD2("After Reset", game.state()->to_string(false, false));
      int game_outcome = game.run(show_game);
      if(game_outcome == 1)
         stats0.add_win("flag", game.state()->get_turn_count());
      else if(game_outcome == 2)
         stats0.add_win("moves", game.state()->get_turn_count());
      else if(game_outcome == -1)
         stats1.add_win("flag", game.state()->get_turn_count());
      else if(game_outcome == -2)
         stats1.add_win("moves", game.state()->get_turn_count());
      else {
         stats0.add_draw();
         stats1.add_draw();
      }
      LOGD2("After game played", game.state()->to_string(false, false));
      if(sim % print_every_n_sim == 0) {
         print_round_results(
            sim, num_sims, *game.get_agent_0(), *game.get_agent_1(), stats0, stats1);
      }
   }
   std::cout << std::endl;
   return std::make_tuple(stats0, stats1);
}

}  // namespace aze::arena