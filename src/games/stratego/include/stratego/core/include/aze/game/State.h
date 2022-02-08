

#pragma once

#include <unordered_set>
#include <variant>

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

   [[nodiscard]] inline size_t turn_count() { return m_turn_count; }

  private:
   board_type m_board;

   Status m_status;
   bool m_status_checked;
   size_t m_turn_count;

   history_type m_move_history;

   utils::random::RNG m_rng;

  protected:
   inline bool &status_checked() { return m_status_checked; }
   inline void incr_turn_count(size_t amount = 1) { m_turn_count += amount; }
   virtual State *clone_impl() const = 0;

  public:
   explicit State(
      board_type board,
      size_t turn_count = 0,
      const history_type &history = {},
      std::optional< std::variant< size_t, utils::random::RNG > > seed = std::nullopt);

   State(
      board_type board,
      std::optional< std::variant< size_t, utils::random::RNG > > seed);

   virtual ~State() = default;

   virtual void apply_action(const action_type &action) = 0;
   virtual Status check_terminal() = 0;
   virtual void restore_to_round(size_t round);

   [[nodiscard]] virtual Team active_team() const = 0;

   sptr< State > clone() const { return sptr< State >(clone_impl()); }

   void undo_last_rounds(size_t n = 1);

   inline auto &rng() { return m_rng; }
   inline auto &board() { return m_board; }

   [[nodiscard]] inline auto rng() const { return m_rng; }
   [[nodiscard]] inline auto turn_count() const { return m_turn_count; }
   Status status()
   {
      if(m_status_checked)
         return m_status;
      LOGD("Checking terminality.")
      m_status_checked = true;
      m_status = check_terminal();
      return m_status;
   }
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
   [[nodiscard]] virtual std::string to_string() const = 0;
   [[nodiscard]] virtual std::string to_string(Team team, bool hide_unknowns) const = 0;
};

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
void State< BoardType, HistoryType, PieceType, Action >::undo_last_rounds(size_t n)
{
   for(size_t i = 0; i < n; ++i) {
      auto [turn, team, move, pieces] = m_move_history.pop_last();
      m_board[move[1]] = std::move(std::get<1>(pieces));
      m_board[move[0]] = std::move(std::get<0>(pieces));
   }
   m_turn_count -= n;
}

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
void State< BoardType, HistoryType, PieceType, Action >::restore_to_round(size_t round)
{
   if(round > m_turn_count) [[unlikely]] {
      throw std::invalid_argument("Given round is greater than current turn count.");
   } else {
      undo_last_rounds(m_turn_count - round);
   }
}

template < typename BoardType, typename HistoryType, typename PieceType, typename Action >
State< BoardType, HistoryType, PieceType, Action >::State(
   board_type board,
   size_t turn_count,
   const history_type &history,
   std::optional< std::variant< size_t, utils::random::RNG > > seed)
    : m_board(std::move(board)),
      m_status(Status(0)),
      m_status_checked(false),
      m_turn_count(turn_count),
      m_move_history(history),
      m_rng(
         seed.has_value()
            ? std::visit([](auto input) { return utils::random::create_rng(input); }, seed.value())
            : utils::random::create_rng())
{
}
template < class BoardType, class HistoryType, class PieceType, class ActionType >
State< BoardType, HistoryType, PieceType, ActionType >::State(
   board_type board,
   std::optional< std::variant< size_t, utils::random::RNG > > seed)
    : m_board(std::move(board)),
      m_status(Status(0)),
      m_status_checked(false),
      m_turn_count(0),
      m_move_history(),
      m_rng(
         seed.has_value()
            ? std::visit([](auto input) { return utils::random::create_rng(input); }, seed.value())
            : utils::random::create_rng())
{
}

}  // namespace aze