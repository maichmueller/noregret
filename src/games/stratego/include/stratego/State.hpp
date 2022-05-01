
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
   // strong types for turns and indices
   using Turn = fluent::NamedType< size_t, struct TurnTag >;
   using Index = fluent::NamedType< size_t, struct IndexTag >;

   [[nodiscard]] inline auto operator[](Turn turn) const
      -> std::tuple< Team, Action, std::pair< Piece, std::optional< Piece > > >
   {
      return {m_teams.at(turn.get()), m_actions.find(turn.get())->second, m_pieces.at(turn.get())};
   }

   [[nodiscard]] inline auto operator[](Index index) const
      -> std::tuple< Team, Action, std::pair< Piece, std::optional< Piece > > >
   {
      auto turn = m_turns[index.get()];
      return (*this)[Turn(turn)];
   }

   void commit_action(
      size_t turn,
      Team team,
      const Action &action,
      const std::pair< Piece, std::optional< Piece > > &pieces)
   {
      // emplace needed bc 'Action' class not default constructible (requirement for [] operator)
      m_actions.emplace(turn, action);
      // emplace needed bc 'Piece' class not default constructible
      m_pieces.emplace(turn, pieces);
      m_teams[turn] = team;
      m_turns.emplace_back(turn);
   }

   void commit_action(const Board &board, Action action, size_t turn)
   {
      commit_action(turn, Team(turn % 2), action, {board[action[0]].value(), board[action[1]]});
   }

   auto view_team_history(Team team)
   {
      return ranges::views::filter(
         ranges::views::zip(m_turns, m_actions, m_pieces),
         [&, team = team](const auto &common_view) {
            return m_teams[std::get< 0 >(common_view)] == team;
         });
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
   std::map< size_t, std::pair< Piece, std::optional< Piece > > > m_pieces;
};

class State: public aze::State< Board, History, Piece, Action > {
  public:
   using base_type = aze::State< Board, History, Piece, Action >;
   using graveyard_type = std::map< Team, std::map< Token, unsigned int > >;

   template < typename... Params >
   State(Config config, graveyard_type graveyard, uptr< Logic > logic, Params &&... params)
       : base_type(std::forward< Params >(params)...),
         m_config(std::move(config)),
         m_graveyard(std::move(graveyard)),
         m_logic(std::move(logic))
   {
   }

   explicit State(
      Config config,
      std::optional< std::variant< size_t, aze::utils::random::RNG > > seed = std::nullopt);

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
   aze::Status check_terminal() override;
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
