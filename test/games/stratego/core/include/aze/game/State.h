

#pragma once

#include <unordered_set>
#include <variant>

#include "Action.h"
#include "Piece.h"
#include "aze/utils.h"

namespace aze {

template < class BoardType, class HistoryType, class PieceType, class ActionType >
class State {
  public:
   using board_type = BoardType;
   using piece_type = PieceType;
   using position_type = typename piece_type::position_type;
   using action_type = ActionType;
   using history_type = HistoryType;

  private:
   board_type m_board;

   Status m_status;
   bool m_status_checked;
   size_t m_turn_count;

   history_type m_move_history;

   aze::utils::RNG m_rng;

  protected:
   virtual State *clone_impl() const = 0;
   virtual void apply_action(const stratego::Action &move);

  public:
   explicit State(
      board_type board,
      size_t turn_count = 0,
      const history_type &history = {},
      std::optional< std::variant< size_t, utils::RNG > > seed = std::nullopt);

   virtual ~State() = default;

   auto &operator[](const typename piece_type::position_type &position)
   {
      return (*m_board)[position];
   }

   const auto &operator[](const typename piece_type::position_type &position) const
   {
      return (*m_board)[position];
   }

   sptr< State > clone() const { return sptr< State >(clone_impl()); }

   int _apply_action(const action_type &action);

   virtual void restore_to_round(int round);

   void undo_last_rounds(int n = 1);

   inline auto &rng() { return m_rng; }
   inline auto &board() { return m_board; }

   [[nodiscard]] inline auto rng() const { return m_rng; }
   [[nodiscard]] inline int turn_count() const { return m_turn_count; }
   [[nodiscard]] inline Status status() const { return m_status; }
   [[nodiscard]] inline auto history() const { return m_move_history; }
   [[nodiscard]] inline auto &history() { return m_move_history; }
   [[nodiscard]] inline auto board() const { return m_board; }

   inline void board(board_type &&board) { m_board = std::move(board); }
   inline Status status(Status status)
   {
      m_status = status;
      m_status_checked = true;
      return status;
   }
   [[nodiscard]] virtual std::string string_representation() const = 0;
   [[nodiscard]] virtual std::string string_representation(Team team, bool hide_unknowns) const = 0;
};

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
void State< BoardType, HistoryType, PieceType, Action >::apply_action(const stratego::Action &move)
{
   m_board[move[1]] = m_board[move[0]];
   m_board[move[0]] = std::nullopt;
}

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
void State< BoardType, HistoryType, PieceType, Action >::undo_last_rounds(int n)
{
   for(int i = 0; i < n; ++i) {
      auto [turn, team, move, pieces] = m_move_history.pop_last();
      m_board[move[1]] = std::move(pieces[1]);
      m_board[move[0]] = std::move(pieces[0]);
   }
   m_turn_count -= n;
}

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
void State< BoardType, HistoryType, PieceType, Action >::restore_to_round(int round)
{
   undo_last_rounds(m_turn_count - round);
}

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
int State< BoardType, HistoryType, PieceType, Action >::_apply_action(
   const State::action_type &action)
{
   // save all info to the history
   sptr< piece_type > piece_from = (*m_board)[action[0]];
   sptr< piece_type > piece_to = (*m_board)[action[1]];
   // copying the pieces here, bc this way they can be fully restored later on
   // (especially when flags have been altered - needed in undoing last rounds)
   m_move_history.push_back(
      {action,
       {std::make_shared< piece_type >(*piece_from), std::make_shared< piece_type >(*piece_to)}});

   m_status_checked = false;
   m_turn_count += 1;

   return apply_action(action);
}

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
State< BoardType, HistoryType, PieceType, Action >::State(
   board_type board,
   size_t turn_count,
   const history_type &history,
   std::optional< std::variant< size_t, utils::RNG > > seed)
    : m_board(std::move(board)),
      m_status(Status(0)),
      m_status_checked(false),
      m_turn_count(turn_count),
      m_move_history(history),
      m_rng(
         seed.has_value()
            ? std::visit([](auto input) { return utils::create_rng(input); }, seed.value())
            : utils::create_rng())
{
}

}  // namespace aze