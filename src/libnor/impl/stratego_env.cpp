
#include "nor/wrappers/stratego_env.hpp"

#include "nor/utils/string_utils.hpp"
#include "stratego/stratego.hpp"

namespace nor::games::stratego {

double Environment::_status_to_reward(::stratego::Status status, Player player)
{
   switch(status) {
      case Status::ONGOING:
      case Status::TIE: {
         return 0.;
      }
      case Status::WIN_BLUE: {
         return player == Player::alex ? 1. : -1.;
      }
      case Status::WIN_RED: {
         return player == Player::bob ? 1. : -1.;
      }
      default: {
         throw std::logic_error("Unknown Status enum returned.");
      }
   }
}
double Environment::reward(Player player, world_state_type& wstate)
{
   return _status_to_reward(wstate.logic()->check_terminal(wstate), player);
}
bool Environment::is_terminal(world_state_type& wstate)
{
   return wstate.status() != Status::ONGOING;
}
std::vector< Environment::action_type > Environment::actions(
   Player player,
   const world_state_type& wstate) const
{
   return m_logic->valid_actions(wstate, to_team(player));
}

void Environment::transition(world_state_type& worldstate, const action_type& action) const
{
   worldstate.transition(action);
}
void Environment::reset(world_state_type& wstate) const
{
   m_logic->reset(wstate);
}
Environment::observation_type Environment::private_observation(
   Player player,
   const Environment::world_state_type& wstate) const
{
   return observation(wstate, player);
}
Environment::observation_type Environment::private_observation(
   Player player,
   const Environment::action_type& action) const
{
   return action.to_string();
}
Environment::observation_type Environment::public_observation(
   const Environment::world_state_type& wstate) const
{
   return observation(wstate);
}
Environment::observation_type Environment::public_observation(
   const Environment::action_type& action) const
{
   return action.to_string();
}

Environment::Environment(uptr< ::stratego::Logic >&& logic) : m_logic(std::move(logic)) {}

Player Environment::active_player(const Environment::world_state_type& wstate) const
{
   return to_player(wstate.active_team());
}
// std::vector< Environment::action_type > Environment::actions(
//    const Environment::info_state_type& istate) const
//{
//    const std::string line_delimiter = "\n";
//    const std::string segment_delimiter = "|";
//    std::string_view latest_state_obs = istate.history().back().second;
//
//    auto position_parser = [](const std::string_view pos_str) {
//       size_t left_paren = pos_str.find("(");
//       size_t right_paren = pos_str.find(")");
//       std::vector< std::string_view > pos_entries = nor::utils::split(
//          pos_str.substr(left_paren + 1, right_paren - left_paren - 1), ",");
//       return Position{
//          std::stoi(std::string(pos_entries[0])), std::stoi(std::string(pos_entries[1]))};
//    };
//
//    auto visibility_parser = [](std::string_view str) {
//       if(str == "!") {
//          return false;
//       } else if(str == "?") {
//          return true;
//       } else {
//          std::stringstream ss;
//          ss << "Given visibility string ";
//          ss << std::quoted(str);
//          ss << " could not be parsed.";
//          throw std::logic_error(ss.str());
//       }
//    };
//
//    auto token_parser = [](std::string_view str) {
//       if(str == "-") {
//          // If an opponent's hidden unit is given, then just assign it as spy. The token of
//          hidden
//          // opponent units is unused for legal action finding anyway
//          return Token::spy;
//       } else {
//          return stratego::utils::from_string< Token >(str);
//       }
//    };
//    Board board;
//    std::array< Config::setup_t, 2 > setups;
//    std::vector< Position > hole_pos;
//    auto lines = nor::utils::split(latest_state_obs, line_delimiter);
//    auto active_team = stratego::utils::from_string< Team >(lines.front());
//    for(auto line : ranges::span(lines).subspan(2)) {
//       auto piece_infos = nor::utils::split(line, segment_delimiter);
//       Position pos = position_parser(piece_infos[0]);
//       Team team = stratego::utils::from_string< Team >(piece_infos[1]);
//       bool is_hidden = visibility_parser(piece_infos[2]);
//       Token token = token_parser(piece_infos[3]);
//       board[pos] = Piece(team, pos, token, is_hidden);
//       if(token == Token::hole) {
//          hole_pos.emplace_back(pos);
//       } else {
//          setups[static_cast< uint8_t >(team)][pos] = token;
//       }
//    }
//
//    return m_logic->valid_actions(State());
// }

std::string observation(const State& state, std::optional< Player > observing_player)
{
   const std::string delimiter = "|";
   std::stringstream ss;
   // start the infostate with the active player at the state followed by the turn count
   ss << "Active team:" << state.active_team() << "\n";
   ss << "Turn count:" << state.turn_count() << "\n";
   ss << "Board dims:" << (state.config().game_dims | ranges::views::all) << "\n";

   for(auto team : std::set{Team::BLUE, Team::RED}) {
      ss << "Graveyard " << stratego::utils::enum_name(team) << ":"
         << (state.graveyard(team) | ranges::views::keys) << "|"
         << (state.graveyard(team) | ranges::views::values) << "\n";
   }
   ss << "Action History:";
   ss << (state.history().actions() | ranges::views::values) << "\n";
   for(const auto& piece_opt : state.board()) {
      if(not piece_opt.has_value()) {
         continue;
      }
      // put information in string rows of the form:
      // (0,1)|BLUE|!|scout\n
      // (1,3)|BLUE|?|spy\n
      // (3,3)|RED|!|spy\n
      // (2,1)|RED|?|-\n
      // with ! indicating an uncovered piece and ? indicating a hidden piece
      const auto& piece = piece_opt.value();
      const bool hidden = piece.flag_hidden();
      const Team team = piece.team();
      ss << piece.position() << delimiter << team << delimiter << (hidden ? "?" : "!") << delimiter;
      if(hidden and (
            not observing_player.has_value()  // this is the case for a public obs
            or to_player(team) != observing_player.value() // this is the case for a private obs
         )
      ) {
         ss << "-";
      } else {
         ss << piece.token();
      }
      ss << "\n";
   }
   return ss.str();
}

}  // namespace nor::games::stratego