
#include "texas_holdem_poker/state.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "texas_holdem_poker/hand_evaluator.hpp"

namespace texholdem {

namespace {

/// exact equality check for chip amounts (amounts are small dyadic sums in practice)
inline bool feq(double a, double b)
{
   return std::fabs(a - b) <= 1e-9;
}

}  // namespace

State::State(PokerConfig config) : m_config(std::move(config))
{
   m_config.validate();
   const auto n = m_config.n_players;
   m_dealer = 0;
   if(n == 2) {
      // heads-up convention: the button player posts the small blind
      m_sb = 0;
      m_bb = 1;
   } else {
      m_sb = static_cast< int >((0 + 1) % n);
      m_bb = static_cast< int >((0 + 2) % n);
   }
   for(size_t p = 0; p < n; ++p) {
      m_players[p].stack =
         (m_config.starting_stacks.empty() ? m_config.starting_stack : m_config.starting_stacks[p]);
      if(m_players[p].stack < m_config.big_blind) {
         throw std::invalid_argument(
            "Every player's starting stack has to be at least one big blind."
         );
      }
   }
   // post the blinds (blinds are not aggressive wagers: nobody has 'acted' yet). The posted
   // amount has to be computed up-front since it consumes from the very stack it reads.
   auto& sb_record = m_players[as_int(m_sb)];
   const double sb_posted = std::min(m_config.small_blind, sb_record.stack);
   sb_record.stack -= sb_posted;
   sb_record.street_contribution += sb_posted;
   sb_record.total_contribution += sb_posted;
   auto& bb_record = m_players[as_int(m_bb)];
   const double bb_posted = std::min(m_config.big_blind, bb_record.stack);
   bb_record.stack -= bb_posted;
   bb_record.street_contribution += bb_posted;
   bb_record.total_contribution += bb_posted;
   // a blind post which consumes a player's entire stack leaves them all-in
   if(feq(sb_record.stack, 0.)) {
      sb_record.allin = true;
   }
   if(feq(bb_record.stack, 0.)) {
      bb_record.allin = true;
   }

   m_current_total_bet = m_config.big_blind;
   m_last_increment = m_config.big_blind;
   m_last_aggressor = -1;
   m_active_player = Player::chance;  // the hole cards are dealt first
}

double State::_to_call(Player player) const
{
   const auto& record = m_players[as_int(player)];
   return std::max(0., m_current_total_bet - record.street_contribution);
}

double State::_min_raise_to(Player) const
{
   return m_current_total_bet + m_last_increment;
}

double State::_max_raise_to(Player player) const
{
   const auto& record = m_players[as_int(player)];
   return record.street_contribution + record.stack;
}

size_t State::_n_actionable() const
{
   size_t count = 0;
   for(size_t p = 0; p < m_config.n_players; ++p) {
      if(_can_act(m_players[p])) {
         ++count;
      }
   }
   return count;
}

int State::_first_actionable_from(int start) const
{
   const auto n = static_cast< int >(m_config.n_players);
   start %= n;
   if(start < 0) {
      start += n;
   }
   for(int k = 0; k < n; ++k) {
      int seat = (start + k) % n;
      if(_can_act(m_players[as_int(seat)])) {
         return seat;
      }
   }
   return -1;
}

bool State::_round_complete() const
{
   if(m_betting_finished) {
      // during an all-in runout no further betting round can be started
      return true;
   }
   for(const auto& record : m_players | std::views::take(m_config.n_players)) {
      if(not _can_act(record)) {
         continue;
      }
      if(not record.acted or not feq(record.street_contribution, m_current_total_bet)) {
         return false;
      }
   }
   return true;
}

void State::_commit_wager_delta(Player player, double delta)
{
   auto& record = m_players[as_int(player)];
   record.street_contribution += delta;
   record.total_contribution += delta;
   record.stack -= delta;
}

void State::_commit_wager_to(Player player, double target_total)
{
   _commit_wager_delta(player, target_total - m_players[as_int(player)].street_contribution);
}

void State::_apply_call_or_check(Player player)
{
   auto& record = m_players[as_int(player)];
   double due = std::min(_to_call(player), record.stack);
   _commit_wager_delta(player, due);
   record.acted = true;
   if(feq(record.stack, 0.) && not feq(due, 0.)) {
      record.allin = true;
   }
}

std::vector< Player > State::players() const
{
   std::vector< Player > out;
   out.reserve(m_config.n_players);
   for(size_t p = 0; p < m_config.n_players; ++p) {
      out.emplace_back(Player(p));
   }
   return out;
}

std::vector< Player > State::remaining_players() const
{
   std::vector< Player > out;
   for(size_t p = 0; p < m_config.n_players; ++p) {
      if(not m_players[p].folded) {
         out.emplace_back(Player(p));
      }
   }
   return out;
}

std::vector< Card > State::community_cards() const
{
   std::vector< Card > out;
   out.reserve(m_board_dealt);
   for(size_t i = 0; i < m_board_dealt; ++i) {
      out.emplace_back(*m_board[i]);
   }
   return out;
}

double State::pot() const
{
   double total = 0.;
   for(size_t p = 0; p < m_config.n_players; ++p) {
      total += m_players[p].total_contribution;
   }
   return total;
}

bool State::operator==(const State& other) const
{
   if(not (m_config == other.m_config)) {
      return false;
   }
   if(m_dealer != other.m_dealer || m_sb != other.m_sb || m_bb != other.m_bb) {
      return false;
   }
   if(m_active_player != other.m_active_player || m_street != other.m_street) {
      return false;
   }
   if(m_dealt != other.m_dealt || m_holes_dealt != other.m_holes_dealt
      || m_board_dealt != other.m_board_dealt) {
      return false;
   }
   if(m_board != other.m_board || m_holes != other.m_holes) {
      return false;
   }
   if(not feq(m_current_total_bet, other.m_current_total_bet)
      || not feq(m_last_increment, other.m_last_increment)) {
      return false;
   }
   if(m_last_aggressor != other.m_last_aggressor
      || m_betting_finished != other.m_betting_finished) {
      return false;
   }
   if(m_actions.size() != other.m_actions.size()) {
      return false;
   }
   bool players_equal = std::ranges::all_of(
      std::views::zip(m_players, other.m_players),
      [](const auto& zipped) {
         const auto& [mine, theirs] = zipped;
         return mine.folded == theirs.folded and mine.allin == theirs.allin
                and mine.acted == theirs.acted;
      }
   );
   bool chips_equal = std::ranges::all_of(
      std::views::zip(m_players, other.m_players),
      [](const auto& zipped) {
         const auto& [mine, theirs] = zipped;
         return std::fabs(mine.stack - theirs.stack) <= 1e-9
                and std::fabs(mine.street_contribution - theirs.street_contribution) <= 1e-9
                and std::fabs(mine.total_contribution - theirs.total_contribution) <= 1e-9;
      }
   );
   bool history_equal = std::ranges::all_of(
      std::views::zip(m_actions, other.m_actions),
      [](const auto& zipped) { return std::get< 0 >(zipped) == std::get< 1 >(zipped); }
   );
   return players_equal and chips_equal and history_equal;
}

bool State::is_terminal() const
{
   size_t n_live = remaining_players().size();
   if(n_live <= 1) {
      // everyone else folded
      return true;
   }
   if(m_holes_dealt < hole_cards_per_player * m_config.n_players) {
      // still dealing the hole cards
      return false;
   }
   if(m_board_dealt < community_card_count) {
      // more board cards are pending (regular deal or all-in runout)
      return false;
   }
   // full board available: terminal iff the river betting round concluded
   return _round_complete();
}

std::vector< Action > State::_betting_actions(Player player) const
{
   constexpr double tol = 1e-9;
   const double bb = m_config.big_blind;
   const double max_wager = _max_raise_to(player);

   std::vector< Action > out;
   out.reserve(3 + m_config.bet_size_multiples.size());
   out.emplace_back(Action{ActionType::fold});
   if(feq(_to_call(player), 0.)) {
      out.emplace_back(Action{ActionType::check});
   } else {
      out.emplace_back(Action{ActionType::call});
   }

   double largest_candidate = 0.;
   if(feq(m_current_total_bet, 0.)) {
      // no wager on this street yet --> betting
      for(double multiple : m_config.bet_size_multiples) {
         double target = bb * multiple;
         if(target + tol >= bb && target <= max_wager + tol
            && not std::ranges::contains(out, Action{ActionType::bet, target})) {
            largest_candidate = std::max(largest_candidate, target);
            out.emplace_back(Action{ActionType::bet, target});
         }
      }
   } else {
      // facing a wager --> raising ('target' is the raise-to total street contribution)
      double min_target = _min_raise_to(player);
      for(double multiple : m_config.bet_size_multiples) {
         double target = m_current_total_bet + bb * multiple;
         if(target + tol >= min_target && target <= max_wager + tol
            && not std::ranges::contains(out, Action{ActionType::raise, target})) {
            largest_candidate = std::max(largest_candidate, target);
            out.emplace_back(Action{ActionType::raise, target});
         }
      }
   }
   // an explicit jam whenever going all-in adds a wager size beyond every ladder candidate
   // (this includes short-stack jams below the minimum raise size)
   if(max_wager > largest_candidate + tol and max_wager > _to_call(player) + tol) {
      out.emplace_back(Action{ActionType::all_in});
   }
   return out;
}

std::vector< Action > State::actions() const
{
   if(m_active_player == Player::chance or is_terminal()) {
      return {};
   }
   return _betting_actions(m_active_player);
}

std::vector< Card > State::chance_actions() const
{
   if(m_active_player != Player::chance or is_terminal()) {
      return {};
   }
   bool holes_pending = m_holes_dealt < hole_cards_per_player * m_config.n_players;
   bool board_pending = m_board_dealt < community_card_count;
   if(not holes_pending and not board_pending) {
      return {};
   }
   std::vector< Card > out;
   out.reserve(m_config.deck_size - m_dealt.count());
   for(size_t index = 0; index < m_config.deck_size; ++index) {
      if(not m_dealt[index]) {
         out.emplace_back(Card{Rank(2 + index / 4), Suit(index % 4)});
      }
   }
   return out;
}

double State::chance_probability(Card outcome) const
{
   auto outcomes = chance_actions();
   if(std::ranges::contains(outcomes, outcome)) {
      return 1. / static_cast< double >(outcomes.size());
   }
   return 0.;
}

bool State::is_valid(Action action) const
{
   if(m_active_player == Player::chance) {
      return false;
   }
   return std::ranges::contains(actions(), action);
}

bool State::is_valid(Card outcome) const
{
   if(m_active_player != Player::chance or is_terminal() or outcome.index() >= m_config.deck_size
      or is_dealt(outcome)) {
      return false;
   }
   return true;
}

void State::apply_action(Card outcome)
{
   if(not is_valid(outcome)) [[unlikely]] {
      throw std::invalid_argument("The given card was already dealt or no card is due.");
   }
   m_dealt.set(outcome.index());
   const auto n = m_config.n_players;
   if(m_holes_dealt < hole_cards_per_player * n) {
      // deal the next hole card: the first card goes to the small blind, then clockwise; the
      // second card again starting at the small blind
      auto recipient = next_deal_recipient();
      m_holes[as_int(recipient)][m_holes_dealt / n] = outcome;
      ++m_holes_dealt;
      if(m_holes_dealt == hole_cards_per_player * n) {
         // preflop betting starts left of the big blind (heads-up: at the button/small blind)
         int start = (n == 2) ? m_sb : ((m_bb + 1) % static_cast< int >(n));
         m_active_player = Player(_first_actionable_from(start));
      }
   } else {
      assert(m_board_dealt < community_card_count);
      m_board[m_board_dealt++] = outcome;
      if(m_board_dealt == 3 || m_board_dealt == 4 || m_board_dealt == 5) {
         m_street = Street(static_cast< uint8_t >(m_board_dealt - 2));  // 3 -> flop, ...
         if(_n_actionable() >= 2) {
            _start_new_betting_round();
         }
         // otherwise we are in an all-in runout and chance keeps dealing until showdown
      }
   }
}

void State::_start_new_betting_round()
{
   for(auto& record : m_players | std::views::take(m_config.n_players)) {
      record.acted = false;
   }
   m_current_total_bet = 0.;
   m_last_increment = m_config.big_blind;
   m_last_aggressor = -1;
   // postflop action always starts left of the dealer button
   int start = (m_dealer + 1) % static_cast< int >(m_config.n_players);
   m_active_player = Player(_first_actionable_from(start));
}

void State::_conclude_street()
{
   // sweep the street contributions into their totals and prepare the next round
   for(auto& record : m_players | std::views::take(m_config.n_players)) {
      record.street_contribution = 0.;
      record.acted = false;
   }
   m_current_total_bet = 0.;
   m_last_increment = m_config.big_blind;
   m_last_aggressor = -1;
   m_active_player = Player::chance;
   if(_n_actionable() <= 1 or m_street == Street::river) {
      // no further betting action can take place: either the river round concluded (the hand
      // goes to showdown) or every remaining player is all-in (the board runs out)
      m_betting_finished = true;
   }
}

void State::apply_action(Action action)
{
   if(m_active_player == Player::chance) [[unlikely]] {
      throw std::logic_error("Cannot apply a player action while chance is due to act.");
   }
   Player player = m_active_player;
   auto& self = m_players[as_int(player)];

   switch(action.kind) {
      case ActionType::fold: {
         self.folded = true;
         self.acted = true;
         break;
      }
      case ActionType::check:
      case ActionType::call: {
         _apply_call_or_check(player);
         break;
      }
      case ActionType::bet:
      case ActionType::raise: {
         if(not is_valid(action)) [[unlikely]] {
            throw std::invalid_argument(
               "Illegal bet/raise amount " + std::to_string(action.amount) + " for the given state."
            );
         }
         double previous_total = m_current_total_bet;
         _commit_wager_to(player, action.amount);
         m_last_increment = action.amount - previous_total;
         m_current_total_bet = action.amount;
         m_last_aggressor = as_int(player);
         self.acted = true;
         if(feq(self.stack, 0.)) {
            self.allin = true;
         }
         // everyone else gets another chance to respond to the new wager
         for(size_t p = 0; p < m_config.n_players; ++p) {
            if(Player(p) != player and _can_act(m_players[p])) {
               m_players[p].acted = false;
            }
         }
         break;
      }
      case ActionType::all_in: {
         double delta = self.stack;
         double new_total = self.street_contribution + delta;
         _commit_wager_delta(player, delta);
         self.allin = true;
         self.acted = true;
         if(new_total > m_current_total_bet + 1e-9) {
            // this jam is an effective raise (possibly below the minimum raise size)
            m_last_increment = new_total - m_current_total_bet;
            m_current_total_bet = new_total;
            m_last_aggressor = as_int(player);
            for(size_t p = 0; p < m_config.n_players; ++p) {
               if(Player(p) != player and _can_act(m_players[p])) {
                  m_players[p].acted = false;
               }
            }
         }
         break;
      }
      default: throw std::invalid_argument("Unknown action type.");
   }
   m_actions.emplace_back(ActionRecord{player, action, m_street});

   size_t n_live = remaining_players().size();
   if(n_live <= 1) {
      // the last player standing takes the pot without any further cards being dealt
      m_active_player = Player::chance;
      return;
   }
   if(_round_complete()) {
      _conclude_street();
      return;
   }
   m_active_player = Player(_first_actionable_from(as_int(player) + 1));
}

std::vector< std::pair< double, std::vector< Player > > > State::_build_side_pots() const
{
   using PotLayer = std::pair< double, std::vector< Player > >;
   std::vector< PotLayer > pots;
   const auto n = m_config.n_players;
   constexpr double tol = 1e-9;

   double settled = 0.;
   while(true) {
      // find the smallest contribution level above the already settled amount
      double level = std::numeric_limits< double >::max();
      for(size_t p = 0; p < n; ++p) {
         if(m_players[p].total_contribution > settled + tol) {
            level = std::min(level, m_players[p].total_contribution);
         }
      }
      if(level == std::numeric_limits< double >::max()) {
         break;
      }
      double amount = 0.;
      for(size_t p = 0; p < n; ++p) {
         amount += std::min(m_players[p].total_contribution, level) - settled;
      }
      std::vector< Player > eligibles;
      for(size_t p = 0; p < n; ++p) {
         if(not m_players[p].folded and m_players[p].total_contribution >= level - tol) {
            eligibles.emplace_back(Player(p));
         }
      }
      pots.emplace_back(PotLayer{amount, std::move(eligibles)});
      settled = level;
   }
   return pots;
}

uint32_t State::_hand_score(Player player) const
{
   std::array< Card, hole_cards_per_player + community_card_count > cards{};
   size_t used = 0;
   for(const auto& opt_card : m_holes[as_int(player)]) {
      cards[used++] = *opt_card;
   }
   for(size_t i = 0; i < m_board_dealt; ++i) {
      cards[used++] = *m_board[i];
   }
   return eval::evaluate(cards.data(), used);
}

double State::payoff(Player player) const
{
   if(player == Player::chance) {
      throw std::invalid_argument("Can't provide payoff for chance player.");
   }
   if(not is_terminal()) {
      return 0.;
   }
   return payoffs()[as_int(player)];
}

std::vector< double > State::payoffs() const
{
   if(not is_terminal()) {
      return std::vector< double >(m_config.n_players, 0.);
   }
   std::vector< double > winnings(m_config.n_players, 0.);
   auto remaining = remaining_players();
   if(remaining.size() == 1) {
      // winner takes everything (his own uncalled excess is returned implicitly since it was
      // paid by himself)
      winnings[as_int(remaining.front())] = pot();
   } else {
      std::unordered_map< Player, uint32_t > scores;
      for(const auto& [amount, eligibles] : _build_side_pots()) {
         std::vector< Player > layer_winners;
         uint32_t best_score = 0;
         for(auto player : eligibles) {
            uint32_t score;
            auto find_iter = scores.find(player);
            if(find_iter != scores.end()) {
               score = find_iter->second;
            } else {
               score = _hand_score(player);
               scores.emplace(player, score);
            }
            if(layer_winners.empty() or score > best_score) {
               layer_winners.clear();
               layer_winners.emplace_back(player);
               best_score = score;
            } else if(score == best_score) {
               layer_winners.emplace_back(player);
            }
         }
         double share = amount / static_cast< double >(layer_winners.size());
         for(auto winner : layer_winners) {
            winnings[as_int(winner)] += share;
         }
      }
   }
   // rewards are net payoffs: subtract each player's own commitment
   std::vector< double > out(m_config.n_players, 0.);
   for(size_t p = 0; p < m_config.n_players; ++p) {
      out[p] = winnings[p] - m_players[p].total_contribution;
   }
   return out;
}

}  // namespace texholdem
