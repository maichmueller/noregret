
#pragma once

#include <unordered_set>

#include "BoardStratego.h"
#include "LogicStratego.h"
#include <aze/aze.h>

class HistoryStratego {
  public:
   using move_type = BoardStratego::move_type;
   using piece_type = BoardStratego::piece_type;

   inline auto get_by_turn(size_t turn)
      -> std::tuple< Team, move_type, std::array< piece_type, 2 > >
   {
      return {m_teams[turn], m_moves[turn], m_pieces[turn]};
   }
   [[nodiscard]] inline auto get_by_turn(size_t turn) const
      -> std::tuple< Team, move_type, std::array< piece_type, 2 > >
   {
      return {m_teams.at(turn), m_moves.at(turn), m_pieces.at(turn)};
   }
   inline auto get_by_index(size_t index)
      -> std::tuple< Team, move_type, std::array< piece_type, 2 > >
   {
      auto turn = m_turns[index];
      return get_by_turn(turn);
   }
   [[nodiscard]] inline auto get_by_index(size_t index) const
      -> std::tuple< Team, move_type, std::array< piece_type, 2 > >
   {
      auto turn = m_turns[index];
      return get_by_turn(turn);
   }

   void commit_move(
      size_t turn,
      Team team,
      const move_type &move,
      const std::array< piece_type, 2 > &pieces)
   {
      m_moves[turn] = move;
      m_pieces[turn] = pieces;
      m_teams[turn] = Team(turn % 2);
      m_turns[turn] = turn;
   }

   void commit_move(const BoardStratego &board, move_type move, size_t turn)
   {
      commit_move(turn, Team(turn % 2), move, {*board[move[0]], *board[move[1]]});
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
      auto ret = std::tuple{turn, m_teams.at(turn), m_moves.at(turn), m_pieces.at(turn)};
      m_teams.erase(turn);
      m_moves.erase(turn);
      m_pieces.erase(turn);
      return ret;
   }

   [[nodiscard]] auto size() const { return m_turns.size(); }
   [[nodiscard]] auto &turns() const { return m_turns; }
   [[nodiscard]] auto &moves() const { return m_moves; }
   [[nodiscard]] auto &pieces() const { return m_pieces; }
   [[nodiscard]] auto &teams() const { return m_teams; }

  private:
   std::vector< size_t > m_turns;
   std::map< size_t, BoardStratego::move_type > m_moves;
   std::map< size_t, Team > m_teams;
   std::map< size_t, std::array< BoardStratego::piece_type, 2 > > m_pieces;
};

class StateStratego: public State< BoardStratego, HistoryStratego > {
  public:
   using base_type = State< BoardStratego, HistoryStratego >;

   // just decorate all base constructors with initializing also the dead pieces
   // variable.
   template < typename... Params >
   StateStratego(Params &&...params) : base_type(std::forward< Params >(params)...), m_dead_pieces()
   {
   }

   // also declare some explicit constructors
   explicit StateStratego(size_t shape_x, size_t shape_y);

   explicit StateStratego(size_t shape = 5);

   StateStratego(
      size_t shape,
      const std::map< position_type, token_type > &setup_0,
      const std::map< position_type, token_type > &setup_1);

   StateStratego(
      std::array< size_t, 2 > shape,
      const std::map< position_type, token_type > &setup_0,
      const std::map< position_type, token_type > &setup_1);

   StateStratego(
      size_t shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1);

   StateStratego(
      std::array< size_t, 2 > shape,
      const std::map< position_type, int > &setup_0,
      const std::map< position_type, int > &setup_1);

   int _do_move(const move_type &move) override;

  protected:
   static int fight(piece_type &attacker, piece_type &defender);

  private:
   using dead_pieces_type = std::array< std::unordered_set< token_type >, 2 >;
   dead_pieces_type m_dead_pieces;

   void _update_dead_pieces(const sptr< piece_type > &piece)
   {
      if(! piece->is_null())
         m_dead_pieces[piece->get_team()].emplace(piece->get_token());
   }

   StateStratego *clone_impl() const override;
};
