#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "Action.hpp"
#include "Agent.h"
#include "Config.hpp"
#include "Logic.hpp"
#include "State.hpp"
#include "StrategoDefs.hpp"

namespace stratego {

class Game {
  public:
   using agent_type = Agent< State >;
   constexpr static size_t n_teams = 2;

   Game(State &&state, const sptr< agent_type > &ag0, const sptr< agent_type > &ag1)
       : m_state(std::make_unique< State >(std::move(state))), m_agents{ag0, ag1}
   {
   }

   Game(uptr< State > state, const sptr< agent_type > &ag0, const sptr< agent_type > &ag1)
       : m_state(std::move(state)), m_agents{ag0, ag1}
   {
   }
   Game(Game &&) noexcept = default;
   Game &operator=(Game &&) noexcept = default;
   ~Game() = default;

   Status run(const sptr< utils::Plotter< State > > &plotter);
   void reset();

   [[nodiscard]] static constexpr auto nr_players() { return n_teams; }
   auto agents() { return m_agents; }
   auto agent(Team team) { return m_agents.at(static_cast< size_t >(team)); }
   [[nodiscard]] auto &state() const { return *m_state; }
   auto &state() { return *m_state; }

  private:
   uptr< State > m_state;
   std::array< sptr< agent_type >, n_teams > m_agents;
};

}  // namespace stratego
