#include "stratego/Utils.hpp"

#include "common/common.hpp"
#include "stratego/Action.hpp"
#include "stratego/Piece.hpp"

namespace stratego::utils {

std::string print_board(const Board& board, std::optional< Team > team, bool hide_unknowns)
{
   using Team = Team;
#if defined(_MSC_VER)
   static const std::string VERT_BAR{"|"};
   static const std::string RESET;
   static const std::string BLUE;
   static const std::string RED;
#else
   static const std::string VERT_BAR{"\u2588"};
   static const std::string RESET{"\x1B[0m"};
   static const std::string BLUE{"\x1B[44m"};
   static const std::string RED{"\x1B[41m"};
#endif

   int H_SIZE_PER_PIECE = 9;
   int V_SIZE_PER_PIECE = 3;
   // the space needed to assign row indices to the rows and to add a splitting
   // bar "|"
   int row_ind_space = 4;
   int mid = V_SIZE_PER_PIECE / 2;
   const auto& [dim_x, dim_y] = board.shape();
   const auto vert_limit = static_cast< int >(dim_x);
   const auto horiz_limit = static_cast< int >(dim_y);

   // piece string lambda function that returns a str of the token
   // "-1 \n
   // 10.1 \n
   //   1"
   auto create_piece_str = [&H_SIZE_PER_PIECE, &team, &mid, &hide_unknowns](
                              const std::optional< Piece >& piece_opt, int line
                           ) {
      if(not piece_opt.has_value()) {
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
      }
      const auto& piece = piece_opt.value();
      std::string color;
      if(piece.token() == Token::hole) {
#if defined(_MSC_VER)
         // print hole over field
         return utils::center("HOLE", H_SIZE_PER_PIECE, " ");
#else
         // piece is a hole -> return a gray colored field
         return "\x1B[30;47m" + common::center("", H_SIZE_PER_PIECE, " ") + RESET;
#endif
      } else if(piece.team() == Team::RED) {
         color = RED;  // background red, text "white"
      } else {
         color = BLUE;
      }
      if(line == mid - 1) {
         // hidden info line
         return color + common::center(piece.flag_hidden() ? "?" : " ", H_SIZE_PER_PIECE, " ")
                + RESET;
      }
      if(line == mid) {
         // token info line
         if(hide_unknowns and piece.flag_hidden() and team.has_value()
            and piece.team() != team.value()) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + RESET;
         }
         const auto& token = piece.token();
         return color
                + common::center(std::to_string(static_cast< int >(token)), H_SIZE_PER_PIECE, " ")
                + RESET;
      }
      if(line == mid + 1) {
#if defined(_MSC_VER)
         // team info line
         return common::center(piece.team() == Team::BLUE ? "B" : "R", H_SIZE_PER_PIECE, " ");
#else
         // for non msvc we have a colored box to work for us
         return color + common::center("", H_SIZE_PER_PIECE, " ") + RESET;
#endif
      } else {
         // empty line
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
      }
   };

   std::stringstream board_print;
   board_print << "\n";

   std::string init_space = std::string(static_cast< unsigned long >(row_ind_space), ' ');
   std::string h_border = common::repeat(
      VERT_BAR, static_cast< size_t >(horiz_limit * (H_SIZE_PER_PIECE + 1) - 1)
   );

   board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   std::string init = board_print.str();
   std::optional< Piece > curr_piece = std::nullopt;

   // row means row of the board. not actual rows of console output.
   // iterate backwards through the rows for correct display
   //   for(int row = static_cast< int >(dim_y - 1); row > -1; --row) {
   for(int row = vert_limit - 1; row > -1; --row) {
      // per piece we have V_SIZE_PER_PIECE many lines to fill consecutively.
      // Iterate over every column and append the new segment to the right line.
      std::vector< std::stringstream > line_streams(static_cast< unsigned int >(V_SIZE_PER_PIECE));

      for(int col = 0; col < horiz_limit; ++col) {
         if(team == Team::RED) {
            curr_piece = board(
               static_cast< int >(dim_x) - 1 - row, static_cast< int >(dim_y) - 1 - col
            );
         } else {
            curr_piece = board(row, col);
         }

         for(int i = 0; i < V_SIZE_PER_PIECE; ++i) {
            std::stringstream curr_stream;

            if(i == mid - 1 || i == mid + 1) {
               if(col == 0) {
                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space), ' ');
               }
               curr_stream << VERT_BAR << create_piece_str(curr_piece, i);
            } else if(i == mid) {
               if(col == 0) {
                  if(row < 10) {
                     curr_stream << " " << row;
                  } else {
                     curr_stream << row;
                  }
                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space - 2), ' ')
                              << VERT_BAR;
               }
               curr_stream << create_piece_str(curr_piece, i);
               if(col != horiz_limit - 1) {
                  curr_stream << VERT_BAR;
               }
            }
            // extend the current line i by the new information
            line_streams[size_t(i)] << curr_stream.str();
         }
      }
      for(auto& stream : line_streams) {
         board_print << stream.str() << VERT_BAR << "\n";
      }
      board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   }
   // column width for the row index plus vertical dash
   board_print << std::string(static_cast< unsigned long >(row_ind_space), ' ');
   // print the column index rows
   for(int i = 0; i < horiz_limit; ++i) {
      board_print << common::center(std::to_string(i), H_SIZE_PER_PIECE + 1, " ");
   }
   board_print << "\n";
   return board_print.str();
}

constexpr common::CEBijection< Status, std::string_view, 4 > status_name_bij = {
   std::pair{Status::TIE, "TIE"},
   std::pair{Status::WIN_RED, "WIN_RED"},
   std::pair{Status::WIN_BLUE, "WIN_BLUE"},
   std::pair{Status::ONGOING, "ONGOING"},
};
constexpr common::CEBijection< Token, std::string_view, 13 > token_name_bij = {
   std::pair{Token::flag, "Flag"},
   std::pair{Token::spy, "spy"},
   std::pair{Token::scout, "scout"},
   std::pair{Token::miner, "miner"},
   std::pair{Token::sergeant, "sergeant"},
   std::pair{Token::lieutenant, "lieutenant"},
   std::pair{Token::captain, "captain"},
   std::pair{Token::major, "major"},
   std::pair{Token::colonel, "colonel"},
   std::pair{Token::general, "general"},
   std::pair{Token::marshall, "marshall"},
   std::pair{Token::bomb, "bomb"},
   std::pair{Token::hole, "hole"},
};
constexpr common::CEBijection< Team, std::string_view, 3 > team_name_bij = {
   std::pair{Team::BLUE, "BLUE"},
   std::pair{Team::RED, "RED"},
   std::pair{Team::NEUTRAL, "NEUTRAL"},
};
constexpr common::CEBijection< FightOutcome, std::string_view, 3 > fightoutcome_name_bij = {
   std::pair{FightOutcome::kill, "kill"},
   std::pair{FightOutcome::death, "death"},
   std::pair{FightOutcome::tie, "tie"},
};
constexpr common::CEBijection< DefinedBoardSizes, std::string_view, 3 > definedboardsizes_name_bij =
   {
      std::pair{DefinedBoardSizes::small, "small"},
      std::pair{DefinedBoardSizes::medium, "medium"},
      std::pair{DefinedBoardSizes::large, "large"},
};
}  // namespace stratego::utils

namespace common {

template <>
std::string to_string(const stratego::Status& e)
{
   return std::string(stratego::utils::status_name_bij.at(e));
}
template <>
std::string to_string(const stratego::Team& e)
{
   return std::string(stratego::utils::team_name_bij.at(e));
}
template <>
std::string to_string(const stratego::FightOutcome& e)
{
   return std::string(stratego::utils::fightoutcome_name_bij.at(e));
}
template <>
std::string to_string(const stratego::Token& e)
{
   return std::string(stratego::utils::token_name_bij.at(e));
}
template <>
std::string to_string(const stratego::DefinedBoardSizes& e)
{
   return std::string(stratego::utils::definedboardsizes_name_bij.at(e));
}
template <>
std::string to_string(const stratego::Action& e)
{
   return e.to_string();
}
template <>
auto from_string(std::string_view str) -> stratego::Status
{
   return stratego::utils::status_name_bij.at(str);
}
template <>
auto from_string(std::string_view str) -> stratego::Team
{
   return stratego::utils::team_name_bij.at(str);
}
template <>
auto from_string(std::string_view str) -> stratego::FightOutcome
{
   return stratego::utils::fightoutcome_name_bij.at(str);
}
template <>
auto from_string(std::string_view str) -> stratego::Token
{
   return stratego::utils::token_name_bij.at(str);
}
template <>
auto from_string(std::string_view str) -> stratego::DefinedBoardSizes
{
   return stratego::utils::definedboardsizes_name_bij.at(str);
}

}  // namespace common
