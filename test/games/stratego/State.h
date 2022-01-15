
#pragma once

#include <aze/aze.h>

#include <unordered_set>
#include <utility>

#include "Config.hpp"
#include "Logic.h"
#include "StrategoDefs.hpp"

namespace stratego {

class History {
   using Team = aze::Team;

  public:
   ;

   inline auto get_by_turn(size_t turn) -> std::tuple< Team, Action, std::array< Piece, 2 > >
   {
      return {m_teams[turn], m_actions[turn], m_pieces[turn]};
   }
   [[nodiscard]] inline auto get_by_turn(size_t turn) const
      -> std::tuple< Team, Action, std::array< Piece, 2 > >
   {
      return {m_teams.at(turn), m_actions.at(turn), m_pieces.at(turn)};
   }
   inline auto get_by_index(size_t index) -> std::tuple< Team, Action, std::array< Piece, 2 > >
   {
      auto turn = m_turns[index];
      return get_by_turn(turn);
   }
   [[nodiscard]] inline auto get_by_index(size_t index) const
      -> std::tuple< Team, Action, std::array< Piece, 2 > >
   {
      auto turn = m_turns[index];
      return get_by_turn(turn);
   }

   void
   commit_action(size_t turn, Team team, const Action &action, const std::array< Piece, 2 > &pieces)
   {
      m_actions[turn] = action;
      m_pieces[turn] = pieces;
      m_teams[turn] = team;
      m_turns[turn] = turn;
   }

   void commit_action(const Board &board, Action action, size_t turn)
   {
      Board b;
      b(Position{0,0});
      commit_action(turn, Team(turn % 2), action, {board(action[0]), board(action[1])});
   }

   auto pop_last()
   {
      /**
       * @brief Remove the latest entries from the history. Return the
       * contents, that were removed.
       * @return tuple,
       *  all removed entries in sequence: turn, team, move, pieces
       */
      auto turn = m_turns.back();
      auto ret = std::tuple{turn, m_teams.at(turn), m_actions.at(turn), m_pieces.at(turn)};
      m_teams.erase(turn);
      m_actions.erase(turn);
      m_pieces.erase(turn);
      return ret;
   }

   [[nodiscard]] auto size() const { return m_turns.size(); }
   [[nodiscard]] auto &turns() const { return m_turns; }
   [[nodiscard]] auto &actions() const { return m_actions; }
   [[nodiscard]] auto &pieces() const { return m_pieces; }
   [[nodiscard]] auto &teams() const { return m_teams; }

  private:
   std::vector< size_t > m_turns;
   std::map< size_t, Action > m_actions;
   std::map< size_t, Team > m_teams;
   std::map< size_t, std::array< Piece, 2 > > m_pieces;
};

class State: public aze::State< Board, History, Piece, Action > {
  public:
   using base_type = aze::State< Board, History, Piece, Action >;
   using graveyard_type = std::array< std::unordered_set< Piece::token_type >, 2 >;

   template < typename... Params >
   State(Config config, Params &&...params)
       : base_type(std::forward< Params >(params)...), m_config(std::move(config)), m_graveyard()
   {
   }

   explicit State(Config config);

   int apply_action(const action_type &action) override;

   [[nodiscard]] inline auto &config() const { return m_config; }
   [[nodiscard]] inline auto graveyard() const { return m_graveyard; }
   [[nodiscard]] inline auto graveyard(int team) const { return m_graveyard[team]; }

  protected:
   static int fight(piece_type &attacker, piece_type &defender);

  private:
   /// the specific configuration of the stratego game belonging to this state
   Config m_config;
   /// the graveyard of dead pieces
   graveyard_type m_graveyard;

   void _to_graveyard(const std::optional< piece_type > &piece_opt)
   {
      if(! piece_opt.has_value())
         m_graveyard[static_cast<int>(piece_opt.value().team())].emplace(piece_opt.value().token());
   }

   [[nodiscard]] State *clone_impl() const override;
};

}  // namespace stratego
