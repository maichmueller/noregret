
#pragma once

#include <named_type.hpp>
#include <unordered_set>
#include <utility>

#include "Action.hpp"
#include "Config.hpp"
#include "Piece.hpp"
#include "StrategoDefs.hpp"

namespace stratego {

class Logic;

class History {
  public:
   auto begin() { return m_history_element.begin(); }
   auto begin() const { return m_history_element.begin(); }
   auto end() { return m_history_element.end(); }
   auto end() const { return m_history_element.end(); }

   [[nodiscard]] auto &operator[](size_t turn) const { return m_history_element.at(turn); }

   void commit_action(
      size_t turn,
      Team team,
      Action action,
      std::pair< Piece, std::optional< Piece > > pieces
   )
   {
      m_history_element.emplace(turn, std::tuple{team, std::move(action), std::move(pieces)});
      m_turns.emplace_back(turn);
   }

   void commit_action(const Board &board, Action action, size_t turn)
   {
      commit_action(turn, Team(turn % 2), action, {board[action[0]].value(), board[action[1]]});
   }

   auto view_team_history(Team team)
   {
      return ranges::views::filter(m_history_element, [&, team = team](const auto &elem_pair_view) {
         const auto &[team_, action, pieces] = std::get< 1 >(elem_pair_view);
         return team_ == team;
      });
   }

   auto pop_last()
   {
      /**
       * @brief Remove the latest entries from the private_history. Return the
       * contents, that were removed.
       * @return tuple,
       *  all removed entries in sequence: turn, team, move, pieces
       */
      auto turn = m_turns.back();
      auto elem_iter = m_history_element.find(turn);
      auto ret = std::tuple_cat(std::forward_as_tuple(turn), std::move(elem_iter->second));
      m_turns.pop_back();
      m_history_element.erase(elem_iter);
      return ret;
   }

   auto view_last()
   {
      /**
       * @brief Remove the latest entries from the private_history. Return the
       * contents, that were removed.
       * @return tuple,
       *  all removed entries in sequence: turn, team, move, pieces
       */
      if(m_turns.empty()) {
         return m_history_element.end();
      }
      auto turn = m_turns.back();
      auto ret = m_history_element.find(turn);
      return ret;
   }

   [[nodiscard]] auto size() const { return m_turns.size(); }
   [[nodiscard]] auto &turns() const { return m_turns; }
   [[nodiscard]] auto &elements_map() const { return m_history_element; }

  private:
   std::vector< size_t > m_turns{};
   std::unordered_map<
      size_t,
      std::tuple< Team, Action, std::pair< Piece, std::optional< Piece > > > >
      m_history_element{};
};

class State {
  public:
   using graveyard_type = std::map< Team, std::map< Token, unsigned int > >;

  private:
   /// the specific configuration of the stratego game belonging to this state
   Config m_config;
   /// the board of pieces to play on
   Board m_board;
   /// the graveyard of dead pieces
   graveyard_type m_graveyard;
   /// the currently used game logic on this state
   uptr< Logic > m_logic;

   Status m_status;
   bool m_status_checked;
   size_t m_turn_count;

   History m_move_history;

   common::RNG m_rng;

   bool &status_checked() { return m_status_checked; }
   void incr_turn_count(size_t amount = 1) { m_turn_count += amount; }

   [[nodiscard]] State *clone_impl() const;

   void _fill_dead_pieces();

  public:
   State(
      Config config,
      graveyard_type graveyard,
      uptr< Logic > logic,
      Board board,
      size_t turn_count = 0,
      const History &history = {},
      std::optional< std::variant< size_t, common::RNG > > seed = std::nullopt
   );

   explicit
   State(Config config, std::optional< std::variant< size_t, common::RNG > > seed = std::nullopt);

   // definitions for these needs to be in .cpp due to Logic being an incomplete type here and
   // uniq_ptr accessing it in its destructor
   State(const State &);
   State &operator=(const State &state);
   State(State &&) noexcept = default;
   State &operator=(State &&) noexcept = default;
   ~State() = default;

   void transition(const Action &action);
   Status check_terminal();
   void restore_to_round(size_t round);

   void undo_last_rounds(size_t n = 1);

   inline auto &rng() { return m_rng; }
   inline auto &board() { return m_board; }

   [[nodiscard]] auto rng() const { return m_rng; }
   [[nodiscard]] auto turn_count() const { return m_turn_count; }
   Status status()
   {
      if(m_status_checked) {
         return m_status;
      }
      SPDLOG_DEBUG("Checking terminality.");
      m_status_checked = true;
      m_status = check_terminal();
      return m_status;
   }
   [[nodiscard]] auto history() const { return m_move_history; }
   [[nodiscard]] auto &history() { return m_move_history; }
   [[nodiscard]] auto board() const { return m_board; }

   void board(Board &&board) { m_board = std::move(board); }
   Status status(Status status)
   {
      m_status = status;
      m_status_checked = true;
      return status;
   }
   [[nodiscard]] std::string to_string() const;
   [[nodiscard]] std::string to_string(std::optional< Team > team, bool hide_unknowns)
      const;

   void to_graveyard(const std::optional< Piece > &piece_opt)
   {
      if(piece_opt.has_value()) {
         m_graveyard[piece_opt.value().team()][piece_opt.value().token()]++;
      }
   }

   void transition(Move move);
   [[nodiscard]] Team active_team() const
   {
      return Team((turn_count() + static_cast< size_t >(m_config.starting_team)) % 2);
   }

   [[nodiscard]] auto &config() const { return m_config; }
   [[nodiscard]] auto *logic() const { return &*m_logic; }
   [[nodiscard]] auto &graveyard() const { return m_graveyard; }
   [[nodiscard]] auto &graveyard(Team team) const { return m_graveyard.at(team); }
};

}  // namespace stratego
