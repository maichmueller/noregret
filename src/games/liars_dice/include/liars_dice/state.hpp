
#ifndef NOR_LIARS_DICE_STATE_HPP
#define NOR_LIARS_DICE_STATE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace liars_dice {

/// the maximal number of players this game supports
constexpr size_t max_player_count = 4;

enum class Player : int8_t { chance = -1, one = 0, two = 1, three = 2, four = 3 };

template < std::integral To = size_t, typename T >
inline To as_int(T p)
{
   // we let things silently fail in the call site if Player::chance is passed in here for example
   return static_cast< To >(p);
}

/**
 * @brief Configuration of a (multi-round, multiplayer) liar's dice game.
 *
 * 'n_players' seats participate; each starts with 'dice_per_player' unbiased dice with
 * 'n_faces' faces each (faces are the integers 1..n_faces). The benchmark instantiation of
 * the papers uses n_faces in [3, 6].
 *
 * The default configuration {2 players x 1 die} together with the single-round flow of a
 * challenge-decided game reproduces the original 2-player single-die benchmark exactly.
 */
struct DiceConfig {
   uint8_t n_players = 2;  //< number of participating seats, has to be within [2, 4]
   uint8_t dice_per_player = 1;  //< starting dice per player, >= 1
   uint8_t n_faces = 6;  //< number of faces per die, has to be within [3, 6]

   DiceConfig() = default;
   /// backwards-compatible single-argument construction: only the face count is set
   explicit DiceConfig(uint8_t n_faces_) : n_faces(n_faces_) {}
   DiceConfig(uint8_t n_players_, uint8_t dice_per_player_, uint8_t n_faces_)
       : n_players(n_players_), dice_per_player(dice_per_player_), n_faces(n_faces_)
   {
   }

   void validate() const
   {
      if(n_players < 2 or n_players > max_player_count) {
         throw std::invalid_argument(
            "liar's dice supports between 2 and 4 players (got " + std::to_string(n_players) + ")."
         );
      }
      if(dice_per_player < 1) {
         throw std::invalid_argument(
            "liar's dice requires at least one die per player (got "
            + std::to_string(dice_per_player) + ")."
         );
      }
      if(n_faces < 3 or n_faces > 6) {
         throw std::invalid_argument(
            "liar's dice supports between 3 and 6 die faces (got " + std::to_string(n_faces) + ")."
         );
      }
   }

   /// the total number of dice in play at game start
   uint8_t total_dice() const { return uint8_t(n_players * dice_per_player); }

   /// the highest legal bid count at game start: every die could show the bid face
   uint8_t max_bid_count() const { return total_dice(); }

   friend bool operator==(const DiceConfig&, const DiceConfig&) = default;
};

/**
 * @brief A chance outcome: the hidden roll of one of a player's dice.
 *
 * 'slot' indexes the die within the player's current holding (always 0 while every player
 * holds a single die, i.e. in the default configuration).
 */
struct Roll {
   Player player;
   uint8_t face;  //< in [1, n_faces]
   uint8_t slot = 0;

   friend bool operator==(const Roll&, const Roll&) = default;
};

/// a bid claim of 'count' dice showing 'face' across ALL alive players' dice
struct Bid {
   uint8_t count = 1;  //< in [1, number of alive dice]
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
 * @brief A world state of a multiplayer elimination liar's dice match.
 *
 * Game flow:
 * - each round, every alive player rolls all his remaining dice (chance deals seat order among
 *   alive players, uniform per die face).
 * - bidding proceeds cyclically among the alive players. The round opener rotates per round:
 *   the nominal opener of round k is seat (k mod n_players), advanced clockwise to the next
 *   alive seat if that seat is already eliminated (round 0 therefore opens at seat 0 like the
 *   original single-round game and OpenSpiel's post-deal opener rule).
 * - a bid {count, face} claims 'count' dice showing 'face' among all ALIVE dice; a raise must
 *   strictly dominate the standing bid lexicographically. Facing the top bid
 *   {alive_dice, n_faces} a challenge is the only legal action.
 * - a challenge reveals all alive dice publicly. If actual count >= claimed count the bidder
 *   wins the round (the challenger loses one die), else the challenger wins (the bidder loses
 *   one die). A player whose last die is lost is eliminated.
 * - the match is terminal once a single player remains standing; he wins.
 *
 * Payoff convention (zero-sum): winner +1, each other player -1/(n_players - 1). For the
 * default 2-player configuration this degenerates to the original +/-1 scoring.
 *
 * world_state = (config, dice[n_players][dice_per_player], dice_left[n_players], round,
 *                current_bid, actions, active_player, round_outcomes). Compact, cheap to copy,
 *                strongly comparable.
 */
class State {
  public:
   explicit State(DiceConfig config = {}) : m_config(config)
   {
      m_config.validate();
      m_dice.assign(m_config.n_players, std::vector< uint8_t >(m_config.dice_per_player, 0));
      m_rolled.assign(m_config.n_players, 0);
      m_dice_left.assign(m_config.n_players, m_config.dice_per_player);
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
         Player
            loser = bidder_holds ? m_active_player /* challenger */ : _last_bidder() /* bidder */;
         m_latest_outcome = bidder_holds ? Outcome::bidder_wins : Outcome::challenger_wins;
         m_round_outcomes.emplace_back(*m_latest_outcome);
         m_dice_left[as_int(loser)] -= 1;
         if(_alive_count() > 1) {
            _start_next_round();
         }
         return;
      }
      m_actions.emplace_back(m_active_player, action);
      m_current_bid = action.bid;
      m_active_player = _next_alive_seat(m_active_player);
   }

   void apply_action(Roll outcome)
   {
      if(not is_valid(outcome)) {
         throw std::logic_error("Invalid liar's dice chance outcome.");
      }
      auto seat = as_int(outcome.player);
      m_dice[seat][outcome.slot] = outcome.face;
      m_rolled[seat] += 1;
      m_roll_history.emplace_back(outcome);
      if(_all_alive_dice_rolled()) {
         // all dice dealt --> the round opener starts the bidding
         m_active_player = opener();
      } else {
         m_active_player = _next_roller();
      }
   }

   /// the match ends once a single player remains standing
   [[nodiscard]] bool is_terminal() const { return _alive_count() == 1; }
   [[nodiscard]] bool is_valid(Action action) const;
   [[nodiscard]] bool is_valid(Roll outcome) const;
   [[nodiscard]] std::vector< Action > actions() const;
   [[nodiscard]] std::vector< Roll > chance_actions() const;
   [[nodiscard]] double chance_probability(Roll outcome) const;

   /**
    * payoff of `player` at this state under the zero-sum convention
    * (+1 for the last-standing winner / evenly split -1 for everyone else; 0 while running).
    * The default 2-player configuration yields the original +/-1 scoring.
    */
   [[nodiscard]] double payoff(Player player) const
   {
      if(player == Player::chance) {
         throw std::invalid_argument("Can't provide payoff for chance player.");
      }
      if(not is_terminal()) {
         return 0.;
      }
      if(alive(player)) {
         return 1.;
      }
      return -1. / double(m_config.n_players - 1);
   }

   /// payoffs in seat order
   [[nodiscard]] std::vector< double > payoffs() const
   {
      std::vector< double > out;
      out.reserve(m_config.n_players);
      for(uint8_t seat = 0; seat < m_config.n_players; ++seat) {
         out.emplace_back(payoff(Player(seat)));
      }
      return out;
   }

   ///////////////////////
   /// API: accessors  ///
   ///////////////////////

   [[nodiscard]] Player active_player() const { return m_active_player; }
   [[nodiscard]] std::vector< Player > players() const
   {
      std::vector< Player > out;
      out.reserve(m_config.n_players + 1);
      out.emplace_back(Player::chance);
      for(uint8_t seat = 0; seat < m_config.n_players; ++seat) {
         out.emplace_back(Player(seat));
      }
      return out;
   }
   [[nodiscard]] const auto& config() const { return m_config; }
   /// own first die face of `player` (empty before it was rolled or once eliminated)
   [[nodiscard]] std::optional< uint8_t > die(Player player) const
   {
      auto seat = as_int(player);
      if(not alive(player) or m_rolled[seat] == 0) {
         return std::nullopt;
      }
      return std::optional< uint8_t >(m_dice[seat][0]);
   }
   /// the rolled faces of `player`'s dice in the running round (empty while unrolled/dead)
   [[nodiscard]] std::vector< uint8_t > dice(Player player) const
   {
      auto seat = as_int(player);
      std::vector< uint8_t > out;
      if(alive(player)) {
         out.reserve(m_rolled[seat]);
         for(uint8_t slot = 0; slot < m_rolled[seat]; ++slot) {
            out.emplace_back(m_dice[seat][slot]);
         }
      }
      return out;
   }
   /// how many dice have been rolled so far in the running round
   [[nodiscard]] size_t rolls_done() const
   {
      size_t total = 0;
      for(auto count : m_rolled) {
         total += count;
      }
      return total;
   }
   /// the currently standing bid (empty before the first bid of the running round)
   [[nodiscard]] const std::optional< Bid >& current_bid() const { return m_current_bid; }
   /// the player who made the currently standing bid
   [[nodiscard]] Player current_bidder() const { return _last_bidder(); }
   /// public history of betting actions taken so far, across all rounds (in order)
   [[nodiscard]] const auto& actions_history() const { return m_actions; }
   /// chronological history of all chance rolls ever applied, across all rounds
   [[nodiscard]] const auto& roll_history() const { return m_roll_history; }
   /// per-round resolutions so far (one entry per completed challenge)
   [[nodiscard]] const auto& round_outcomes() const { return m_round_outcomes; }
   /// the resolution of the most recent challenge (empty until the first challenge was made)
   [[nodiscard]] const std::optional< Outcome >& challenge_outcome() const
   {
      return m_latest_outcome;
   }
   /// number of dice among all alive players' dice showing `face`
   [[nodiscard]] uint8_t actual_count(uint8_t face) const
   {
      uint8_t count = 0;
      for(auto seat : _alive_seats()) {
         for(uint8_t slot = 0; slot < m_rolled[seat]; ++slot) {
            count += uint8_t(m_dice[seat][slot] == face);
         }
      }
      return count;
   }
   /// whether `player` still holds at least one die
   [[nodiscard]] bool alive(Player player) const { return m_dice_left[as_int(player)] != 0; }
   /// how many dice `player` currently holds
   [[nodiscard]] uint8_t dice_left(Player player) const { return m_dice_left[as_int(player)]; }
   /// the zero-based round index (the first roll-and-bid cycle is round 0)
   [[nodiscard]] size_t round_index() const { return m_round; }
   /// the seat that opens (or opened) the bidding in the running round
   [[nodiscard]] Player opener() const
   {
      Player nominal = Player(m_round % m_config.n_players);
      if(alive(nominal)) {
         return nominal;
      }
      return _next_alive_seat(nominal);
   }
   /// the number of dice currently held across all alive players (upper bid bound)
   [[nodiscard]] uint8_t max_bid_count() const
   {
      return uint8_t(std::accumulate(m_dice_left.begin(), m_dice_left.end(), 0));
   }
   /// strong equality over all state fields
   bool operator==(const State& other) const = default;

  private:
   [[nodiscard]] std::vector< uint8_t > _alive_seats() const
   {
      std::vector< uint8_t > out;
      out.reserve(m_config.n_players);
      for(uint8_t seat = 0; seat < m_config.n_players; ++seat) {
         if(m_dice_left[seat] != 0) {
            out.emplace_back(seat);
         }
      }
      return out;
   }
   [[nodiscard]] size_t _alive_count() const { return _alive_seats().size(); }
   [[nodiscard]] bool _all_alive_dice_rolled() const
   {
      for(auto seat : _alive_seats()) {
         if(m_rolled[seat] < m_dice_left[seat]) {
            return false;
         }
      }
      return true;
   }
   /// the next alive seat clockwise of `after`
   [[nodiscard]] Player _next_alive_seat(Player after) const
   {
      auto current = as_int(after);
      for(uint8_t step = 1; step <= m_config.n_players; ++step) {
         auto candidate = (current + step) % m_config.n_players;
         if(m_dice_left[candidate] != 0) {
            return Player(candidate);
         }
      }
      return after;
   }
   /// the seat that rolls next during the chance phase (alive seats in order, slots ascending)
   [[nodiscard]] Player _next_roller() const
   {
      for(auto seat : _alive_seats()) {
         if(m_rolled[seat] < m_dice_left[seat]) {
            return Player(seat);
         }
      }
      return m_active_player;
   }
   /// resets all per-round bookkeeping and hands over to the chance player for the re-roll
   void _start_next_round()
   {
      m_round += 1;
      m_current_bid.reset();
      for(auto seat : _alive_seats()) {
         m_rolled[seat] = 0;
      }
      for(auto& row : m_dice) {
         std::fill(row.begin(), row.end(), 0);
      }
      m_active_player = Player::chance;
   }
   /// the player who made the currently standing bid (chance while no bid exists)
   [[nodiscard]] Player _last_bidder() const
   {
      return m_actions.empty() ? Player::chance : m_actions.back().player;
   }

   DiceConfig m_config;
   /// die faces in [1, n_faces] of the running round, 0 marks an unrolled die
   std::vector< std::vector< uint8_t > > m_dice{};
   /// how many dice each seat rolled in the running round
   std::vector< uint8_t > m_rolled{};
   /// remaining dice per seat (0 marks an eliminated player); public knowledge
   std::vector< uint8_t > m_dice_left{};
   size_t m_round = 0;
   std::optional< Bid > m_current_bid{};
   std::vector< ActionRecord > m_actions{};
   std::vector< Roll > m_roll_history{};
   std::vector< Outcome > m_round_outcomes{};
   std::optional< Outcome > m_latest_outcome{};
   Player m_active_player = Player::chance;
};

inline bool State::is_valid(Action action) const
{
   if(is_terminal()) {
      return false;
   }
   if(not _all_alive_dice_rolled()) {
      return false;
   }
   switch(action.kind) {
      case ActionType::challenge:
         // challenging is only possible against a standing bid
         return m_current_bid.has_value();
      case ActionType::bid: {
         const auto& bid = action.bid;
         if(bid.count < 1 || bid.count > max_bid_count()) {
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
   if(is_terminal() || _all_alive_dice_rolled() || m_current_bid.has_value()) {
      return false;
   }
   if(outcome.player != _next_roller()) {
      return false;
   }
   if(as_int(outcome.player) < 0 or as_int< int >(outcome.player) >= int(m_config.n_players)) {
      return false;
   }
   auto seat = as_int(outcome.player);
   if(outcome.slot != m_rolled[seat]) {
      return false;
   }
   return outcome.face >= 1 && outcome.face <= m_config.n_faces;
}

inline std::vector< Action > State::actions() const
{
   std::vector< Action > out;
   if(is_terminal() || not _all_alive_dice_rolled()) {
      return out;
   }
   // bids in lexicographic order (count ascending, then face ascending), ...
   for(uint8_t count = 1; count <= max_bid_count(); ++count) {
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
   if(is_terminal() || _all_alive_dice_rolled() || m_current_bid.has_value()) {
      return {};
   }
   Player roller = _next_roller();
   auto seat = as_int(roller);
   std::vector< Roll > out;
   out.reserve(m_config.n_faces);
   for(uint8_t face = 1; face <= m_config.n_faces; ++face) {
      out.emplace_back(Roll{roller, face, m_rolled[seat]});
   }
   return out;
}

inline double State::chance_probability(Roll outcome) const
{
   if(is_terminal() || _all_alive_dice_rolled() || not is_valid(outcome)) {
      return 0.;
   }
   return 1. / double(m_config.n_faces);
}

}  // namespace liars_dice

#endif  // NOR_LIARS_DICE_STATE_HPP
