
#pragma once

#include <named_type.hpp>
#include <unordered_set>
#include <utility>

#include "Action.hpp"
#include "Config.hpp"
#include "StrategoDefs.hpp"
#include "aze/aze.h"

namespace stratego {

class Logic;

class History {
   using Team = aze::Team;

  public:
   auto begin() { return m_history_element.begin(); }
   auto begin() const { return m_history_element.begin(); }
   auto end() { return m_history_element.end(); }
   auto end() const { return m_history_element.end(); }

   [[nodiscard]] inline auto &operator[](size_t turn) const { return m_history_element.at(turn); }

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
   std::vector< size_t > m_turns;
   std::unordered_map<
      size_t,
      std::tuple< Team, Action, std::pair< Piece, std::optional< Piece > > > >
      m_history_element;
};

class State: public aze::State< Board, History, Piece, Action > {
  public:
   using base_type = aze::State< Board, History, Piece, Action >;
   using graveyard_type = std::map< Team, std::map< Token, unsigned int > >;

   template < typename... Params >
   State(Config config, graveyard_type graveyard, uptr< Logic > logic, Params &&...params)
       : base_type(std::forward< Params >(params)...),
         m_config(std::move(config)),
         m_graveyard(std::move(graveyard)),
         m_logic(std::move(logic))
   {
   }

   explicit State(
      Config config,
      std::optional< std::variant< size_t, aze::utils::random::RNG > > seed = std::nullopt
   );

   // definitions for these needs to be in .cpp due to Logic being an incomplete type here and
   // uniq_ptr accessing it in its destructor
   State(const State &);
   State &operator=(const State &);
   State(State &&) noexcept;
   State &operator=(State &&) noexcept;
   ~State() override;

   void to_graveyard(const std::optional< piece_type > &piece_opt)
   {
      if(piece_opt.has_value()) {
         m_graveyard[piece_opt.value().team()][piece_opt.value().token()]++;
      }
   }

   void transition(const action_type &action) override;
   void transition(Move move);
   aze::Status check_terminal() { return status(std::as_const(*this).check_terminal()); }
   aze::Status check_terminal() const override;
   [[nodiscard]] Team active_team() const override
   {
      return Team((turn_count() + static_cast< size_t >(m_config.starting_team)) % 2);
   }

   [[nodiscard]] std::string to_string() const override;
   [[nodiscard]] std::string to_string(std::optional< Team > team, bool hide_unknowns)
      const override;

   [[nodiscard]] inline auto &config() const { return m_config; }
   [[nodiscard]] inline auto *logic() const { return &*m_logic; }
   [[nodiscard]] inline auto &graveyard() const { return m_graveyard; }
   [[nodiscard]] inline auto &graveyard(Team team) const { return m_graveyard.at(team); }

  private:
   /// the specific configuration of the stratego game belonging to this state
   Config m_config;
   /// the graveyard of dead pieces
   graveyard_type m_graveyard;
   /// the currently used game logic on this state
   uptr< Logic > m_logic;

   [[nodiscard]] State *clone_impl() const override;

   void _fill_dead_pieces();
};

}  // namespace stratego
