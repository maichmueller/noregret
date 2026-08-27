
#ifndef NOR_SHERIFF_STATE_HPP
#define NOR_SHERIFF_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace sheriff {

/// 'none' marks the absence of an acting player (terminal states); its value mirrors
/// nor::Player::unknown so the FOSG adapter's cast stays consistent (cf. oshi_zumo / shapley).
enum class Player : int8_t { none = -2, one = 0, two = 1 };

[[nodiscard]] constexpr Player opponent(Player player)
{
   return player == Player::one ? Player::two : Player::one;
}

template < typename To = size_t, typename T >
[[nodiscard]] constexpr To as_int(T value)
{
   return static_cast< To >(value);
}

// #####################################################################################################################
// rule transcription notes (canonical literature source)
// #####################################################################################################################
//
// SOURCE. Farina, Ling, Fang, Sandholm, "Correlation in Extensive-Form Games: Saddle-Point
// Formulation and Benchmarks", NeurIPS 2019 (arXiv:1905.12564), Section 5.2 and Appendix F.1 --
// the paper that introduced this game as one of the two canonical GENERAL-SUM imperfect-
// information benchmarks for the extensive-form-correlated-equilibrium literature.
//
// PLAYERS. Player::one is the SMUGGLER, Player::two is the SHERIFF (paper: the smuggler acts
// first, Appendix F.1 "At the beginning of the game, the Smuggler loads n items").
//
// PROGRESSION. The smuggler secretly loads n in {0..n_max} items. Then follow r >= 1 rounds of
// bargaining; round t = 1..r comprises the public bribe offer b_t in {0..b_max} followed by the
// sheriff's public Yes/No response. All actions are public knowledge except the cargo selection,
// which only the smuggler knows. Only the FINAL round's offer/response is consequential; the
// first r-1 rounds exist for coordination/signalling (Appendix F.1 and F.2 'passcode' analysis).
//
// PAYOFFS (Appendix F.1, outcomes 1.-3.; v = item value, p = per-item fine, s = false-alarm
// compensation):
// - sheriff accepts b_r:            smuggler gets n*v - b_r,  sheriff gets +b_r
// - sheriff rejects & inspects,
//   goods found (n > 0):           smuggler gets -n*p,       sheriff gets +n*p
// - sheriff rejects & inspects,
//   nothing found (n == 0):        smuggler gets +s,         sheriff gets -s
// The rejection branch always triggers an inspection (the main-text Section 5.2 phrasing
// "if the Sheriff decides not to inspect" corresponds to accepting a zero bribe). The payoffs do
// NOT sum to a constant in general -- the defining property of this general-sum benchmark.
//
// REFERENCE INSTANCES. Baseline instance (Section 5.2): v=5, p=1, s=1, n_max=10, b_max=2, r=2.
// Small instances of Table 3 / F.2: same v,p,s with n_max in {1,2,3,5,10}, b_max=2, r in {1..4}.

/**
 * @brief The phase of the sequentialized game.
 */
enum class Phase : uint8_t { load = 0, offer, respond, over };

/// how a game instance ended (mirrors the three payoff outcomes above)
enum class TerminalCause : uint8_t { none = 0, bribe_accepted, inspection_goods, inspection_clean };

/**
 * @brief Configuration of a Sheriff instance (Farina et al. 2019, Appendix F.1).
 *
 * Defaults transcribe the paper's *baseline* instance of Section 5.2:
 * v = 5, p = 1, s = 1, n_max = 10, b_max = 2, r = 2.
 */
struct Config {
   /// hard cap so the fixed-size bargaining log stays bounded (paper experiments use r <= 4)
   static constexpr size_t max_rounds = 16;

   double v = 5.;  //< value of every illegal item
   double p = 1.;  //< fine per discovered illegal item
   double s = 1.;  //< compensation the sheriff pays on a false alarm
   size_t n_max = 10;  //< maximal number of smuggled items
   size_t b_max = 2;  //< maximal bribe amount
   size_t rounds = 2;  //< number of bargaining rounds r

   Config() = default;
   Config(double v_, double p_, double s_, size_t n_max_, size_t b_max_, size_t rounds_)
       : v(v_), p(p_), s(s_), n_max(n_max_), b_max(b_max_), rounds(rounds_)
   {
      validate();
   }

   void validate() const
   {
      if(v < 0. or p < 0. or s < 0.) {
         throw std::invalid_argument("sheriff parameters v, p, s must be non-negative.");
      }
      if(n_max == 0) {
         throw std::invalid_argument("sheriff requires at least one loadable item (n_max >= 1).");
      }
      if(rounds == 0 or rounds > max_rounds) {
         throw std::invalid_argument(
            "sheriff supports 1 to " + std::to_string(max_rounds) + " bargaining rounds."
         );
      }
   }

   [[nodiscard]] friend bool operator==(const Config&, const Config&) = default;
};

/// the chronological record of one resolved bargaining round (history reconstruction)
struct RoundRecord {
   uint32_t bribe = 0;
   bool accepted = false;

   friend bool operator==(const RoundRecord&, const RoundRecord&) = default;
   friend bool operator!=(const RoundRecord&, const RoundRecord&) = default;
};

/**
 * @brief The actions of the game, tagged by the acting role.
 *
 * Load: the smuggler secretly commits his cargo size. Offer: a bribe amount (public). Respond:
 * the sheriff's accept/reject decision (public).
 */
struct Load {
   uint32_t items = 0;

   friend bool operator==(const Load&, const Load&) = default;
   friend bool operator!=(const Load&, const Load&) = default;
};

struct Offer {
   uint32_t bribe = 0;

   friend bool operator==(const Offer&, const Offer&) = default;
   friend bool operator!=(const Offer&, const Offer&) = default;
};

struct Respond {
   bool accept = false;

   friend bool operator==(const Respond&, const Respond&) = default;
   friend bool operator!=(const Respond&, const Respond&) = default;
};

using Action = std::variant< Load, Offer, Respond >;

/**
 * @brief World state of one Sheriff instance.
 *
 * All data is small and fixed-size (beyond the config) so copying stays cheap. The smuggler's
 * cargo is the only hidden piece of information; it lives here in the world state but never
 * reaches any observation of the sheriff before an inspection terminal.
 */
class State {
  public:
   explicit State(Config config = {}) : m_config(config) { m_config.validate(); }

   ////////////////////////////////
   /// API: transitions        ///
   ////////////////////////////////

   void apply_action(const Action& action)
   {
      if(terminal()) {
         throw std::logic_error("sheriff state is terminal; no further actions can be applied.");
      }
      switch(m_phase) {
         case Phase::load: {
            const auto* load = std::get_if< Load >(&action);
            if(load == nullptr) {
               throw std::invalid_argument("sheriff loading phase only accepts 'Load' actions.");
            }
            if(not is_valid(action)) {
               throw std::invalid_argument(
                  "sheriff: illegal cargo size " + std::to_string(load->items) + "."
               );
            }
            m_cargo = load->items;
            m_phase = Phase::offer;
            m_active = Player::one;
            return;
         }
         case Phase::offer: {
            const auto* offer = std::get_if< Offer >(&action);
            if(offer == nullptr) {
               throw std::invalid_argument("sheriff offer phases only accept 'Offer' actions.");
            }
            if(not is_valid(action)) {
               throw std::invalid_argument(
                  "sheriff: illegal bribe amount " + std::to_string(offer->bribe) + "."
               );
            }
            m_pending_bribe = offer->bribe;
            m_phase = Phase::respond;
            m_active = Player::two;
            return;
         }
         case Phase::respond: {
            const auto* respond = std::get_if< Respond >(&action);
            if(respond == nullptr) {
               throw std::invalid_argument("sheriff response phases only accept 'Respond' actions."
               );
            }
            _resolve(respond->accept);
            return;
         }
         case Phase::over: throw std::logic_error("sheriff: unreachable phase.");
      }
      throw std::logic_error("sheriff: unreachable phase.");
   }

   [[nodiscard]] bool is_valid(const Action& action) const
   {
      if(terminal()) {
         return false;
      }
      switch(m_phase) {
         case Phase::load: {
            const auto* load = std::get_if< Load >(&action);
            return load != nullptr && load->items <= m_config.n_max;
         }
         case Phase::offer: {
            const auto* offer = std::get_if< Offer >(&action);
            return offer != nullptr && offer->bribe <= m_config.b_max;
         }
         case Phase::respond: {
            return std::holds_alternative< Respond >(action);
         }
         case Phase::over: return false;
      }
      return false;
   }

   /// legal actions of whoever is due to act (empty when terminal or not their turn)
   [[nodiscard]] std::vector< Action > actions(Player player) const
   {
      std::vector< Action > out;
      if(terminal() or m_active != player) {
         return out;
      }
      switch(m_phase) {
         case Phase::load: {
            out.reserve(m_config.n_max + 1);
            for(size_t n : std::views::iota(size_t{0}, m_config.n_max + 1)) {
               out.emplace_back(Load{uint32_t(n)});
            }
            break;
         }
         case Phase::offer: {
            out.reserve(m_config.b_max + 1);
            for(size_t b : std::views::iota(size_t{0}, m_config.b_max + 1)) {
               out.emplace_back(Offer{uint32_t(b)});
            }
            break;
         }
         case Phase::respond: {
            out.emplace_back(Respond{true});
            out.emplace_back(Respond{false});
            break;
         }
         case Phase::over: break;
      }
      return out;
   }

   ////////////////////////////////
   /// API: queries            ///
   ////////////////////////////////

   [[nodiscard]] bool terminal() const { return m_phase == Phase::over; }

   /**
    * @brief GENERAL-SUM terminal payoffs exactly as transcribed from Appendix F.1.
    *
    * u(one) = smuggler utility, u(two) = sheriff utility; 0 before terminality. The payoffs
    * deliberately do NOT sum to a constant in general.
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(not terminal()) {
         return 0.;
      }
      const double n = double(*m_cargo);
      const double bribe = double(m_log.at(m_config.rounds - 1).bribe);
      switch(m_terminal_cause) {
         case TerminalCause::bribe_accepted: {
            return player == Player::one ? n * m_config.v - bribe : bribe;
         }
         case TerminalCause::inspection_goods: {
            return player == Player::one ? -n * m_config.p : n * m_config.p;
         }
         case TerminalCause::inspection_clean: {
            return player == Player::one ? m_config.s : -m_config.s;
         }
         case TerminalCause::none: return 0.;
      }
      return 0.;
   }

   [[nodiscard]] std::array< double, 2 > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   [[nodiscard]] Player active_player() const { return m_active; }
   [[nodiscard]] Phase phase() const { return m_phase; }
   [[nodiscard]] const Config& config() const { return m_config; }
   [[nodiscard]] TerminalCause terminal_cause() const { return m_terminal_cause; }
   /// the smuggler's secret cargo (nullopt before the loading action)
   [[nodiscard]] std::optional< uint32_t > cargo() const { return m_cargo; }
   /// the bribe offered in the current (not yet answered) round; nullopt otherwise
   [[nodiscard]] std::optional< uint32_t > pending_bribe() const { return m_pending_bribe; }
   /// number of fully resolved bargaining rounds (each resolved round = offer + response)
   [[nodiscard]] size_t resolved_rounds() const { return m_resolved_rounds; }
   /// the 1-based index of the ongoing bargaining round
   [[nodiscard]] size_t round() const { return m_resolved_rounds + 1; }
   /// the per-round records; entries at indices >= resolved_rounds() are not finalized yet
   [[nodiscard]] const std::array< RoundRecord, Config::max_rounds >& rounds_log() const
   {
      return m_log;
   }

   [[nodiscard]] bool operator==(const State& other) const
   {
      return m_config == other.m_config && m_phase == other.m_phase && m_active == other.m_active
             && m_cargo == other.m_cargo && m_pending_bribe == other.m_pending_bribe
             && m_resolved_rounds == other.m_resolved_rounds && m_log == other.m_log
             && m_terminal_cause == other.m_terminal_cause;
   }
   [[nodiscard]] bool operator!=(const State& other) const { return not (*this == other); }

  private:
   /////////////////////////////////
   /// API: private utilities    ///
   /////////////////////////////////

   /// finalizes round t; intermediate rounds loop back to the next offer, the last one resolves
   void _resolve(bool accept)
   {
      auto& record = m_log.at(m_resolved_rounds);
      record.bribe = *m_pending_bribe;
      record.accepted = accept;
      m_resolved_rounds += 1;
      if(m_resolved_rounds < m_config.rounds) {
         // non-consequential round: communication only, proceed to the next offer
         m_pending_bribe.reset();
         m_phase = Phase::offer;
         m_active = Player::one;
         return;
      }
      // the final response decides the outcome
      m_terminal_cause = [&] {
         if(accept) {
            return TerminalCause::bribe_accepted;
         }
         return *m_cargo > 0 ? TerminalCause::inspection_goods : TerminalCause::inspection_clean;
      }();
      m_phase = Phase::over;
      m_active = Player::none;
   }

   /////////////////////////////////
   /// API: data members         ///
   /////////////////////////////////

   Config m_config{};
   Phase m_phase = Phase::load;
   Player m_active = Player::one;
   std::optional< uint32_t > m_cargo{};
   std::optional< uint32_t > m_pending_bribe{};
   size_t m_resolved_rounds = 0;
   std::array< RoundRecord, Config::max_rounds > m_log{};
   TerminalCause m_terminal_cause = TerminalCause::none;
};

}  // namespace sheriff

#endif  // NOR_SHERIFF_STATE_HPP
