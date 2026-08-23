
#ifndef NOR_LIARS_DICE_STATE_HPP
#define NOR_LIARS_DICE_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace liars_dice {

/// the maximal number of players this game supports
constexpr size_t max_player_count = 2;
/// each player rolls exactly one die
constexpr size_t dice_per_player = 2;

enum class Player : int8_t { chance = -1, one = 0, two = 1 };

template < std::integral To = size_t, typename T >
inline To as_int(T p)
{
   // we let things silently fail in the call site if Player::chance is passed in here for example
   return static_cast< To >(p);
}

/**
 * @brief Configuration of a single-round liar's dice game.
 *
 * Both players roll one unbiased die with 'n_faces' faces (faces are the integers 1..n_faces).
 * The benchmark instantiation of the papers uses n_faces in [3, 6].
 */
struct DiceConfig {
   uint8_t n_faces = 6;  //< number of faces per die, has to be within [3, 6]

   DiceConfig() = default;
   explicit DiceConfig(uint8_t n_faces_) : n_faces(n_faces_) {}

   void validate() const
   {
      if(n_faces < 3 or n_faces > 6) {
         throw std::invalid_argument(
            "liar's dice supports between 3 and 6 die faces (got " + std::to_string(n_faces) + ")."
         );
      }
   }

   /// the highest legal bid count: every die in play could show the bid face
   uint8_t max_bid_count() const { return uint8_t(dice_per_player); }

   friend bool operator==(const DiceConfig&, const DiceConfig&) = default;
};

/**
 * @brief A chance outcome: the hidden roll of player's die.
 */
struct Roll {
   Player player;
   uint8_t face;  //< in [1, n_faces]

   friend bool operator==(const Roll&, const Roll&) = default;
};

/// a bid claim of 'count' dice showing 'face' across both players' dice
struct Bid {
   uint8_t count = 1;  //< in [1, max_bid_count] (i.e. [1, 2])
   uint8_t face = 1;  //< in [1, n_faces]

   /// lexicographic strict dominance: (c',f') > (c,f) iff c'>c or (c'==c and f'>f)
   [[nodiscard]] bool dominates(const Bid& other) const
   {
      return count > other.count || (count == other.count && face > other.face);
   }

   friend bool operator==(const Bid&, const Bid&) = default;
};

enum class ActionType : uint8_t { bid = 0, challenge = 1 };

enum class Outcome : uint8_t { bidder_wins = 0, challenger_wins = 1 };

/**
 * @brief A betting action.
 *
 * Either a bid announcement ('bid' carries its payload, mirroring texas hold'em's tagged
 * action struct) or a challenge of the currently standing bid.
 */
struct Action {
   ActionType kind = ActionType::bid;
   Bid bid{1, 1};  //< ignored when kind == challenge

   friend bool operator==(const Action&, const Action&) = default;
};

/// an applied betting action together with the acting player
struct ActionRecord {
   Player player;
   Action action;

   friend bool operator==(const ActionRecord&, const ActionRecord&) = default;
};

/**
 * @brief A world state of a single-round liar's dice hand.
 *
 * world_state = (dice[2], current_bid, active_player, challenge_resolved). Compact, cheap to copy,
 * strongly comparable.
 */
class State {
  public:
   explicit State(DiceConfig config = {}) : m_config(config)
   {
      m_config.validate();
      m_dice.fill(0);
   }

   ////////////////////////////////////
   /// API: transitions and queries ///
   ////////////////////////////////////

   void apply_action(Action action)
   {
      if(not is_valid(action)) {
         throw std::logic_error("Invalid liar's dice action.");
      }
      if(action.kind == ActionType::challenge) {
         auto standing = *m_current_bid;
         auto actual = actual_count(standing.face);
         bool bidder_holds = actual >= standing.count;
         m_challenge_outcome = bidder_holds ? Outcome::bidder_wins : Outcome::challenger_wins;
         return;
      }
      m_actions.emplace_back(m_active_player, action);
      m_current_bid = action.bid;
      m_active_player = other(m_active_player);
   }

   void apply_action(Roll outcome)
   {
      if(not is_valid(outcome)) {
         throw std::logic_error("Invalid liar's dice chance outcome.");
      }
      m_dice[as_int(outcome.player)] = outcome.face;
      if(_all_dice_rolled()) {
         // both dice dealt --> player one opens the bidding
         m_active_player = Player::one;
      }
   }

   [[nodiscard]] bool is_terminal() const { return m_challenge_outcome.has_value(); }
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(Roll outcome) const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< Roll > chance_actions() const;
   [[nodiscard]] double chance_probability(Roll outcome) const;

   /// payoff of `player` at this state (+1 winner / -1 loser, zero-sum; 0 while running)
   [[nodiscard]] double payoff(Player player) const
   {
      if(player == Player::chance) {
         throw std::invalid_argument("Can't provide payoff for chance player.");
      }
      if(not is_terminal()) {
         return 0.;
      }
      bool player_is_last_bidder = _last_bidder() == player;
      bool player_wins = (*m_challenge_outcome == Outcome::bidder_wins) == player_is_last_bidder;
      return player_wins ? 1. : -1.;
   }

   /// payoffs in seat order {player one, player two}
   [[nodiscard]] std::vector< double > payoffs() const
   {
      return {payoff(Player::one), payoff(Player::two)};
   }

   ///////////////////////
   /// API: accessors  ///
   ///////////////////////

   [[nodiscard]] Player active_player() const { return m_active_player; }
   [[nodiscard]] std::vector< Player > players() const
   {
      return {Player::chance, Player::one, Player::two};
   }
   [[nodiscard]] const auto& config() const { return m_config; }
   /// own die face of `player` (empty before it was rolled)
   [[nodiscard]] std::optional< uint8_t > die(Player player) const
   {
      return m_dice[as_int(player)] == 0 ? std::nullopt
                                         : std::optional< uint8_t >(m_dice[as_int(player)]);
   }
   /// how many dice have been rolled so far
   [[nodiscard]] size_t rolls_done() const
   {
      return _all_dice_rolled() ? 2u : size_t(m_dice[0] != 0);
   }
   /// the currently standing bid (empty before the first bid)
   [[nodiscard]] const std::optional< Bid >& current_bid() const { return m_current_bid; }
   /// the player who made the currently standing bid
   [[nodiscard]] Player current_bidder() const { return _last_bidder(); }
   /// public history of betting actions taken so far (in order)
   [[nodiscard]] const auto& actions_history() const { return m_actions; }
   /// the terminal resolution (empty until a challenge was made)
   [[nodiscard]] const std::optional< Outcome >& challenge_outcome() const
   {
      return m_challenge_outcome;
   }
   /// number of dice among both players' dice showing `face`
   [[nodiscard]] uint8_t actual_count(uint8_t face) const
   {
      return uint8_t((m_dice[0] == face) + (m_dice[1] == face));
   }
   /// strong equality over all state fields
   bool operator==(const State& other) const = default;

  private:
   [[nodiscard]] static Player other(Player p)
   {
      return p == Player::one ? Player::two : Player::one;
   }
   [[nodiscard]] bool _all_dice_rolled() const { return m_dice[0] != 0 && m_dice[1] != 0; }
   /// the player who made the currently standing bid (chance while no bid exists)
   [[nodiscard]] Player _last_bidder() const
   {
      return m_actions.empty() ? Player::chance : m_actions.back().player;
   }

   DiceConfig m_config;
   /// die faces in [1, n_faces], 0 marks an unrolled die
   std::array< uint8_t, dice_per_player > m_dice{};
   std::optional< Bid > m_current_bid{};
   std::vector< ActionRecord > m_actions{};
   Player m_active_player = Player::chance;
   std::optional< Outcome > m_challenge_outcome{};
};

inline bool State::is_valid(Action action) const
{
   if(is_terminal()) {
      return false;
   }
   if(not _all_dice_rolled()) {
      return false;
   }
   switch(action.kind) {
      case ActionType::challenge:
         // challenging is only possible against a standing bid
         return m_current_bid.has_value();
      case ActionType::bid: {
         const auto& bid = action.bid;
         if(bid.count < 1 || bid.count > m_config.max_bid_count()) {
            return false;
         }
         if(bid.face < 1 || bid.face > m_config.n_faces) {
            return false;
         }
         // a raise must strictly dominate the standing bid; any in-range bid may open
         return not m_current_bid.has_value() || bid.dominates(*m_current_bid);
      }
      default: return false;
   }
}

inline bool State::is_valid(Roll outcome) const
{
   if(is_terminal() || _all_dice_rolled()) {
      return false;
   }
   Player roller = m_dice[0] != 0 ? Player::two : Player::one;
   if(outcome.player != roller) {
      return false;
   }
   return outcome.face >= 1 && outcome.face <= m_config.n_faces;
}

inline std::vector< Action > State::actions() const
{
   std::vector< Action > out;
   if(is_terminal() || not _all_dice_rolled()) {
      return out;
   }
   // bids in lexicographic order (count ascending, then face ascending), ...
   for(uint8_t count = 1; count <= m_config.max_bid_count(); ++count) {
      for(uint8_t face = 1; face <= m_config.n_faces; ++face) {
         Action action{ActionType::bid, Bid{count, face}};
         if(is_valid(action)) {
            out.emplace_back(action);
         }
      }
   }
   // ... then the challenge (if a bid stands)
   if(m_current_bid.has_value()) {
      out.emplace_back(Action{ActionType::challenge, Bid{}});
   }
   return out;
}

inline std::vector< Roll > State::chance_actions() const
{
   if(is_terminal() || _all_dice_rolled()) {
      return {};
   }
   Player roller = m_dice[0] != 0 ? Player::two : Player::one;
   std::vector< Roll > out;
   out.reserve(m_config.n_faces);
   for(uint8_t face = 1; face <= m_config.n_faces; ++face) {
      out.emplace_back(Roll{roller, face});
   }
   return out;
}

inline double State::chance_probability(Roll outcome) const
{
   if(is_terminal() || _all_dice_rolled() || not is_valid(outcome)) {
      return 0.;
   }
   return 1. / double(m_config.n_faces);
}

}  // namespace liars_dice

#endif  // NOR_LIARS_DICE_STATE_HPP
