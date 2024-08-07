

#include <gtest/gtest.h>

#include <range/v3/all.hpp>

#include "fixtures.hpp"
#include "testing_utils.hpp"

using namespace stratego;

TEST(State, constructor)
{
   std::map< Position2D, Token > setup0;
   std::map< Position2D, Token > setup1;

   setup0[{0, 0}] = Token::flag;
   setup0[{1, 1}] = Token::scout;

   setup1[{0, 1}] = Token::miner;
   setup1[{1, 0}] = Token::spy;
   std::vector< Position2D > hole_pos;

   auto setup = std::map{
      std::pair{Team::BLUE, std::optional{setup0}}, std::pair{Team::RED, std::optional{setup1}}};

   Config config{Team::BLUE, std::vector{size_t(2), size_t(2)}, setup, hole_pos, false, true, 500};

   State state{config};

   for(auto [pos, token] : setup0) {
      //      std::cout << "Given piece from setup:\n";
      //      std::cout << common::to_string(Team::BLUE) << ", " << pos << ", " << token << "\n";
      //      std::cout << "Board piece:\n";
      //      std::cout << common::to_string(state.board()[pos].value().team()) << ", "
      //                << state.board()[pos].value().position() << ", "
      //                << state.board()[pos].value().token() << "\n";
      EXPECT_EQ((state.board()[pos].value()), Piece(Team::BLUE, pos, token));
   }
   for(auto [pos, token] : setup1) {
      //      std::cout << "Given piece from setup:\n";
      //      std::cout << common::to_string(Team::RED) << ", " << pos << ", " << token << "\n";
      //      std::cout << "Board piece:\n";
      //      std::cout << common::to_string(state.board()[pos].value().team()) << ", "
      //                << state.board()[pos].value().position() << ", "
      //                << state.board()[pos].value().token() << "\n";
      EXPECT_EQ((state.board()[pos].value()), Piece(Team::RED, pos, token));
   }
}

auto get_tokenvector(std::map< Team, std::optional< Config::setup_t > > setups)
{
   std::map< Team, std::map< Token, unsigned int > > counters;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      if(setups.at(team).has_value()) {
         auto values = setups.at(team).value() | ranges::views::values;
         counters[team] = common::counter(std::vector< Token >{values.begin(), values.end()});
      }
   }
   return counters;
}
auto get_tokenpositions(std::map< Team, std::optional< Config::setup_t > > setups)
{
   std::map< Team, std::vector< Position2D > > counters;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      if(setups.at(team).has_value()) {
         auto values = setups.at(team).value() | ranges::views::keys;
         counters[team] = std::vector< Position2D >{values.begin(), values.end()};
      }
   }
   return counters;
}

auto get_tokenvector(std::map< Team, Config::setup_t > setups)
{
   std::map< Team, std::map< Token, unsigned int > > counters;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      auto values = setups.at(team) | ranges::views::values;
      counters[team] = common::counter(std::vector< Token >{values.begin(), values.end()});
   }
   return counters;
}
auto get_tokenpositions(std::map< Team, Config::setup_t > setups)
{
   std::map< Team, std::vector< Position2D > > counters;
   for(auto team : std::set{Team::BLUE, Team::RED}) {
      auto values = setups.at(team) | ranges::views::keys;
      counters[team] = std::vector< Position2D >{values.begin(), values.end()};
   }
   return counters;
}

std::vector< Position2D > get_hole_pos(const Board& board)
{
   std::vector< Position2D > out;
   for(size_t i = 0; i < board.shape(0); i++) {
      for(size_t j = 0; j < board.shape(1); j++) {
         const auto& piece_opt = board[{i, j}];
         if(piece_opt.has_value()) {
            const auto& piece = piece_opt.value();
            if(piece.team() == Team::NEUTRAL && piece.token() == Token::hole) {
               out.emplace_back(int(i), int(j));
            }
         }
      }
   }
   return out;
}

TEST_P(StateConstructorParamsF, constructor_arbitrary_dims)
{
   auto [game_dims, holes, setups] = GetParam();

   auto exp_token_vecs = get_tokenvector(setups);
   auto exp_token_pos_vecs = get_tokenpositions(setups);

   State s(Config(
      starting_team, game_dims, setups, holes, fixed_starting_team, fixed_setups, max_turn_counts
   ));

   //   LOGD2("State", s.to_string())
   auto extracted_setups = Logic::extract_setup(s.board());
   auto obs_token_vecs = get_tokenvector(extracted_setups);
   auto obs_token_pos_vecs = get_tokenpositions(extracted_setups);
   auto obs_hole_pos_vecs = get_hole_pos(s.board());

   EXPECT_EQ(exp_token_vecs[Team::BLUE], obs_token_vecs[Team::BLUE]);
   EXPECT_EQ(exp_token_vecs[Team::RED], obs_token_vecs[Team::RED]);
   EXPECT_EQ(eq_rng(exp_token_pos_vecs[Team::BLUE]), eq_rng(obs_token_pos_vecs[Team::BLUE]));
   EXPECT_EQ(eq_rng(exp_token_pos_vecs[Team::RED]), eq_rng(obs_token_pos_vecs[Team::RED]));
   EXPECT_EQ(eq_rng(holes), eq_rng(obs_hole_pos_vecs));
}

INSTANTIATE_TEST_SUITE_P(
   constructor_arbitrary_dims_tests,
   StateConstructorParamsF,
   ::testing::Values(
      std::tuple{
         std::array< size_t, 2 >{5, 5},
         std::vector< Position2D >{{2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}},
         std::map< Team, std::optional< Config::setup_t > >{
            std::pair{
               Team::BLUE,
               std::optional(Config::setup_t{
                  std::pair{Position2D{0, 0}, Token::flag},
                  std::pair{Position2D{0, 3}, Token::scout},
                  std::pair{Position2D{1, 3}, Token::scout},
                  std::pair{Position2D{0, 1}, Token::major},
                  std::pair{Position2D{1, 1}, Token::bomb}})},
            std::pair{
               Team::RED,
               std::optional(Config::setup_t{
                  std::pair{Position2D{3, 3}, Token::flag},
                  std::pair{Position2D{3, 0}, Token::spy},
                  std::pair{Position2D{4, 0}, Token::spy},
                  std::pair{Position2D{3, 4}, Token::spy}})}}},
      std::tuple{
         std::array< size_t, 2 >{34, 28},
         std::vector< Position2D >{{33, 0}, {0, 10}, {10, 4}, {15, 18}, {20, 12}},
         std::map< Team, std::optional< Config::setup_t > >{
            std::pair{
               Team::BLUE,
               std::optional(Config::setup_t{
                  std::pair{Position2D{0, 27}, Token::flag},
                  std::pair{Position2D{30, 21}, Token::major},
                  std::pair{Position2D{13, 3}, Token::lieutenant},
                  std::pair{Position2D{19, 12}, Token::captain},
                  std::pair{Position2D{1, 1}, Token::major}})},
            std::pair{
               Team::RED,
               std::optional(Config::setup_t{
                  std::pair{Position2D{3, 3}, Token::flag},
                  std::pair{Position2D{9, 7}, Token::flag},
                  std::pair{Position2D{9, 3}, Token::flag},
                  std::pair{Position2D{17, 4}, Token::flag},
                  std::pair{Position2D{7, 16}, Token::bomb}})}}},
      std::tuple{
         std::array< size_t, 2 >{4, 8},
         std::vector< Position2D >{{2, 0}, {3, 7}, {2, 5}, {2, 4}},
         std::map< Team, std::optional< Config::setup_t > >{
            std::pair{
               Team::BLUE,
               std::optional(Config::setup_t{
                  std::pair{Position2D{2, 1}, Token::bomb},
                  std::pair{Position2D{3, 1}, Token::bomb},
                  std::pair{Position2D{1, 1}, Token::bomb},
                  std::pair{Position2D{0, 1}, Token::bomb},
                  std::pair{Position2D{0, 2}, Token::bomb},
                  std::pair{Position2D{0, 3}, Token::bomb},
                  std::pair{Position2D{0, 4}, Token::bomb},
                  std::pair{Position2D{0, 5}, Token::flag}})},
            std::pair{
               Team::RED,
               std::optional(Config::setup_t{
                  std::pair{Position2D{2, 3}, Token::flag},
                  std::pair{Position2D{3, 2}, Token::flag},
                  std::pair{Position2D{3, 3}, Token::flag},
                  std::pair{Position2D{3, 4}, Token::flag},
                  std::pair{Position2D{3, 5}, Token::flag},
                  std::pair{Position2D{3, 6}, Token::flag},
                  std::pair{Position2D{2, 6}, Token::bomb}})}}},
      std::tuple{
         std::array< size_t, 2 >{8, 5},
         std::vector< Position2D >{{6, 0}, {4, 2}, {5, 4}, {2, 3}},
         std::map< Team, std::optional< Config::setup_t > >{
            std::pair{
               Team::BLUE,
               std::optional(Config::setup_t{
                  std::pair{Position2D{2, 1}, Token::spy},
                  std::pair{Position2D{6, 3}, Token::scout},
                  std::pair{Position2D{4, 1}, Token::miner},
                  std::pair{Position2D{5, 1}, Token::major},
                  std::pair{Position2D{6, 1}, Token::lieutenant},
                  std::pair{Position2D{7, 1}, Token::colonel},
                  std::pair{Position2D{7, 2}, Token::captain},
                  std::pair{Position2D{0, 4}, Token::flag}})},
            std::pair{
               Team::RED,
               std::optional(Config::setup_t{
                  std::pair{Position2D{3, 1}, Token::flag},
                  std::pair{Position2D{5, 2}, Token::bomb},
                  std::pair{Position2D{3, 3}, Token::marshall},
                  std::pair{Position2D{3, 4}, Token::general},
                  std::pair{Position2D{4, 4}, Token::colonel},
                  std::pair{Position2D{6, 4}, Token::captain},
                  std::pair{Position2D{7, 4}, Token::spy}})}}}
   )
);
