
#ifndef NOR_FIXTURES_HPP
#define NOR_FIXTURES_HPP

#include <gtest/gtest.h>

#include <stratego/stratego.hpp>

class TinyConfig: public ::testing::Test {
  public:
   using Token = stratego::Token;
   using Team = stratego::Team;
   using Position = stratego::Position2D;
   using Config = stratego::Config;
   using State = stratego::State;

   std::map< Position, Token > setup0;
   std::map< Position, Token > setup1;
   Config cfg;

   TinyConfig() : setup0(), setup1(), cfg(_init_cfg()) {}
   ~TinyConfig() override = default;

   Config _init_cfg()
   {
      // the following setup is build:
      // ----------------
      // |    | 1R | 0R |
      // ----------------
      // |    |    |    |
      // ----------------
      // | 0B | 1B |    |
      // ----------------
      //
      // with a max of 10 turns

      setup0[{0, 0}] = Token::flag;
      setup0[{0, 1}] = Token::spy;
      setup1[{2, 1}] = Token::spy;
      setup1[{2, 2}] = Token::flag;

      Config config{
         Team::BLUE,
         3u,
         std::map{
            std::pair{Team::BLUE, std::make_optional(setup0)},
            std::pair{Team::RED, std::make_optional(setup1)}},
         std::vector< Position >{},
         true,
         true,
         10};

      return config;
   }
};

class StrategoState3x3: public TinyConfig {
  public:
   State state;

   StrategoState3x3() : state(cfg, size_t(0)) {}
   ~StrategoState3x3() override = default;
};

class SmallConfig: public ::testing::Test {
  public:
   using Token = stratego::Token;
   using Team = stratego::Team;
   using Position = stratego::Position2D;
   using Config = stratego::Config;
   using State = stratego::State;

   std::map< Position, Token > setup0;
   std::map< Position, Token > setup1;
   Config cfg;

   SmallConfig() : setup0(), setup1(), cfg(_init_cfg()) {}
   ~SmallConfig() override = default;

   Config _init_cfg()
   {
      setup0[{0, 0}] = Token::flag;
      setup0[{0, 1}] = Token::spy;
      setup0[{0, 2}] = Token::scout;
      setup0[{0, 3}] = Token::scout;
      setup0[{0, 4}] = Token::miner;
      setup0[{1, 0}] = Token::bomb;
      setup0[{1, 1}] = Token::marshall;
      setup0[{1, 2}] = Token::scout;
      setup0[{1, 3}] = Token::bomb;
      setup0[{1, 4}] = Token::miner;
      setup1[{3, 0}] = Token::scout;
      setup1[{3, 1}] = Token::scout;
      setup1[{3, 2}] = Token::bomb;
      setup1[{3, 3}] = Token::scout;
      setup1[{3, 4}] = Token::marshall;
      setup1[{4, 0}] = Token::miner;
      setup1[{4, 1}] = Token::spy;
      setup1[{4, 2}] = Token::bomb;
      setup1[{4, 3}] = Token::miner;
      setup1[{4, 4}] = Token::flag;

      Config config{
         Team::BLUE,
         size_t(5),
         std::map{
            std::pair{Team::BLUE, std::make_optional(setup0)},
            std::pair{Team::RED, std::make_optional(setup1)}},
         std::nullopt,
         true,
         true,
         500};

      return config;
   }
};

class StrategoState5x5: public SmallConfig {
  public:
   State state;

   StrategoState5x5() : state(cfg, size_t(0)) {}
   ~StrategoState5x5() override = default;
};

class BattlematrixParamsF:
    public ::testing::TestWithParam<
       std::tuple< stratego::Token, stratego::Token, stratego::FightOutcome > > {
  protected:
   std::decay_t< decltype(stratego::default_battlematrix()) > bm = stratego::default_battlematrix();
};

class CheckTerminalParamsF:
    public ::testing::TestWithParam< std::tuple<
       unsigned long,
       stratego::Team,
       std::array< size_t, 2 >,
       std::map< stratego::Team, std::optional< stratego::Config::setup_t > >,
       std::map< stratego::Team, std::optional< stratego::Config::token_variant_t > >,
       std::map< stratego::Team, std::optional< std::vector< stratego::Position2D > > >,
       stratego::Status > > {
  protected:
   stratego::Team starting_team = stratego::Team::BLUE;
   bool fixed_starting_team = true;
   size_t max_turn_counts = 1000;
   bool fixed_setups = true;
};

class StateConstructorParamsF:
    public ::testing::TestWithParam< std::tuple<
       std::array< size_t, 2 >,  // game dims
       std::vector< stratego::Position2D >,  // holes
       std::map< stratego::Team, std::optional< stratego::Config::setup_t > >  // setups
       > > {
  protected:
   stratego::Team starting_team = stratego::Team::BLUE;
   bool fixed_starting_team = true;
   size_t max_turn_counts = 1000;
   bool fixed_setups = true;
};

#endif  // NOR_FIXTURES_HPP
