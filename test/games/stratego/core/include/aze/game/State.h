

#pragma once

#include <unordered_set>

#include "Board.h"
#include "Move.h"
#include "Piece.h"
#include "Position.h"
#include "aze/types.h"

namespace aze {

template < class BoardType, class HistoryType >
class State {
  public:
   using board_type = BoardType;
   using piece_type = typename BoardType::piece_type;
   using token_type = typename BoardType::token_type;
   using position_type = typename BoardType::position_type;
   using move_type = Move< position_type >;
   using history_type = HistoryType;
   using graveyard_type = std::array< std::unordered_set< token_type >, 2 >;

  private:
   sptr< board_type > m_board;

   Status m_status;
   bool m_status_checked;
   size_t m_turn_count;

   history_type m_move_history;
   unsigned int m_rounds_without_fight;

   graveyard_type m_graveyard;

   void _recompute_rounds_without_fight();

  protected:
   virtual State *clone_impl() const = 0;
   virtual int _do_move(const move_type &move);

   [[nodiscard]] inline auto &nr_rounds_without_fight() { return m_rounds_without_fight; }

  public:
   template < size_t dim >
   explicit State(
      const std::array< size_t, dim > &shape,
      const std::array< int, dim > &board_starts);

   State(
      sptr< board_type > board,
      Status status,
      bool status_checked = false,
      size_t m_turn_count = 0,
      const history_type &m_history = {},
      unsigned int rounds_without_fight = 0,
      const graveyard_type &graveyard = {});

   virtual ~State() = default;

   auto &operator[](const position_type &position) { return (*m_board)[position]; }

   const auto &operator[](const position_type &position) const { return (*m_board)[position]; }

   sptr< State > clone() const { return sptr< State >(clone_impl()); }

   int do_move(const move_type &move);

   virtual void restore_to_round(int round);

   void undo_last_rounds(int n = 1);

   void move_to_graveyard(int team, sptr< piece_type > piece)
   {
      m_graveyard[team].emplace({piece});
   }

   [[nodiscard]] inline int turn_count() const { return m_turn_count; }
   [[nodiscard]] inline Status status() const { return m_status; }
   [[nodiscard]] inline auto history() const { return m_move_history; }
   [[nodiscard]] inline auto &history() { return m_move_history; }
   [[nodiscard]] inline auto nr_rounds_without_fight() const { return m_rounds_without_fight; }
   [[nodiscard]] inline auto board() const { return m_board; }
   [[nodiscard]] inline auto graveyard() const { return m_graveyard; }
   [[nodiscard]] inline auto graveyard(int team) const { return m_graveyard[team]; }

   inline void set_board(sptr< board_type > brd) { m_board = std::move(brd); }
   inline Status set_status(Status status)
   {
      m_status = status;
      m_status_checked = true;
      return status;
   }
   virtual std::string string_representation(Team team, bool hide_unknowns);
};

template < typename BoardType, typename HistoryType >
int State< BoardType, HistoryType >::_do_move(
   const State< BoardType, HistoryType >::move_type &move)
{
   m_board->update_board(move[1], (*m_board)[move[0]]);
   m_board->update_board(move[0], std::make_shared< piece_type >(move[0]));
   return 0;
}

template < typename BoardType, typename HistoryType >
template < size_t dim >
State< BoardType, HistoryType >::State(
   const std::array< size_t, dim > &shape,
   const std::array< int, dim > &board_starts)
    : State(std::make_shared< board_type >(shape, board_starts))
{
}

template < typename BoardType, typename HistoryType >
void State< BoardType, HistoryType >::undo_last_rounds(int n)
{
   // rwf = rounds without fight
   bool recompute_rwf = false;

   for(int i = 0; i < n; ++i) {
      auto [move, pieces] = m_move_history.back();

      if(m_rounds_without_fight > 0)
         m_rounds_without_fight -= 1;
      else
         recompute_rwf = true;

      m_move_history.pop_back();

      m_board->update_board(move[1], pieces[1]);
      m_board->update_board(move[0], pieces[0]);
   }

   m_turn_count -= n;
   if(recompute_rwf) {
      _recompute_rounds_without_fight();
   }
}

template < typename BoardType, typename HistoryType >
void State< BoardType, HistoryType >::restore_to_round(int round)
{
   undo_last_rounds(m_turn_count - round);
}

template < typename BoardType, typename HistoryType >
int State< BoardType, HistoryType >::do_move(const State::move_type &move)
{
   // save all info to the history
   sptr< piece_type > piece_from = (*m_board)[move[0]];
   sptr< piece_type > piece_to = (*m_board)[move[1]];
   // copying the pieces here, bc this way they can be fully restored later on
   // (especially when flags have been altered - needed in undoing last rounds)
   m_move_history.push_back(
      {move,
       {std::make_shared< piece_type >(*piece_from), std::make_shared< piece_type >(*piece_to)}});

   m_status_checked = false;
   m_turn_count += 1;

   return _do_move(move);
}

template < typename BoardType, typename HistoryType >
std::string State< BoardType, HistoryType >::string_representation(Team team, bool hide_unknowns)
{
   std::stringstream sstream;
   sstream << board()->print_board(team, hide_unknowns) << "\n";
   sstream << "turn count" << m_turn_count << "\n";
   //    sstream << "rounds without fight" << m_rounds_without_fight << "\n";
   return sstream.str();
}
template < typename BoardType, typename HistoryType >
void State< BoardType, HistoryType >::_recompute_rounds_without_fight()
{
   m_rounds_without_fight = 0;
   for(auto hist_rev_iter = m_move_history.rbegin(); hist_rev_iter != m_move_history.rend();
       ++hist_rev_iter) {
      const auto &[_, pieces] = *hist_rev_iter;
      if(not pieces[1]->is_null())
         // if the defending piece was not a null piece, then there was a fight
         break;
      else
         m_rounds_without_fight += 1;
   }
}

template < typename BoardType, typename HistoryType >
State< BoardType, HistoryType >::State(
   sptr< board_type > board,
   Status status,
   bool status_checked,
   size_t turn_count,
   const history_type &m_history,
   unsigned int rounds_without_fight,
   const graveyard_type &graveyard)
    : m_board(std::move(board)),
      m_status(status),
      m_status_checked(status_checked),
      m_turn_count(turn_count),
      m_move_history(m_history),
      m_rounds_without_fight(rounds_without_fight),
      m_graveyard(graveyard)
{
}
}  // namespace aze