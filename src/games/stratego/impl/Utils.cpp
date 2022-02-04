#include "stratego/Utils.hpp"

namespace stratego::utils {

std::string print_board(const Board &board, aze::Team team, bool hide_unknowns)
{
   using Team = aze::Team;
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
   auto [dim_x, dim_y] = board.shape();
   int mid = V_SIZE_PER_PIECE / 2;

   // piece string lambda function that returns a str of the token
   // "-1 \n
   // 10.1 \n
   //   1"
   auto create_piece_str = [&H_SIZE_PER_PIECE, &mid, &hide_unknowns](
                              const std::optional< Piece > &piece_opt, int line) {
      if(not piece_opt.has_value()) {
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
      }
      const auto &piece = piece_opt.value();
      std::string color = BLUE;  // blue by default
      if(piece.token() == Token::hole) {
#if defined(_MSC_VER)
         // print hole over field
         return aze::utils::center("HOLE", H_SIZE_PER_PIECE, " ");
#else
         // piece is a hole -> return a gray colored field
         return "\x1B[30;47m" + aze::utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
#endif
      } else if(piece.team() == Team::RED) {
         color = RED;  // background red, text "white"
      }
      if(line == mid - 1) {
         // hidden info line
         std::string hidden = piece.flag_hidden() ? std::string("?") : std::string(" ");
         return color + aze::utils::center(hidden, H_SIZE_PER_PIECE, " ") + RESET;
      }
      if(line == mid) {
         // type and version info line
         if(hide_unknowns && piece.flag_hidden()) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + RESET;
         }
         const auto &token = piece.token();
         return color
                + aze::utils::center(
                   std::to_string(static_cast< int >(token)), H_SIZE_PER_PIECE, " ")
                + RESET;
      }
      if(line == mid + 1) {
#if defined(_MSC_VER)
         // team info line
         return aze::utils::center(
            piece.team() == aze::Team::BLUE ? "B" : "R", H_SIZE_PER_PIECE, " ");
#else
         // for non msvc we have a colored box to work for us
         return color + aze::utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
#endif
      } else {
         // empty line
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
      }
   };

   std::stringstream board_print;
   board_print << "\n";

   std::string init_space = std::string(static_cast< unsigned long >(row_ind_space), ' ');
   std::string h_border = aze::utils::repeat(
      VERT_BAR, static_cast< unsigned long >(dim_x * (H_SIZE_PER_PIECE + 1) - 1));

   board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   std::string init = board_print.str();
   std::optional< Piece > curr_piece = std::nullopt;

   // row means row of the board. not actual rows of console output.
   // iterate backwards through the rows for correct display
   for(int row = static_cast< int >(dim_y - 1); row > -1; --row) {
      // per piece we have V_SIZE_PER_PIECE many lines to fill consecutively.
      // Iterate over every column and append the new segment to the right line.
      std::vector< std::stringstream > line_streams(static_cast< unsigned int >(V_SIZE_PER_PIECE));

      for(int col = 0; col < dim_x; ++col) {
         // because of the x dimension (horizontal) implicitly defining the column numbers, we have
         // to acces the board pieces in the unintuitive way of (col, row). To be clear, the x
         // coordinate does represent the column, i.e. length in horizontal space, on our board
         // (like in a mathematical 2-d function graph). Likewise, the row implies the length in
         // vertical space.
         if(team == aze::Team::RED) {
            curr_piece = board(dim_x - 1 - col, dim_y - 1 - row);
         } else {
            curr_piece = board(col, row);
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
               if(col != dim_x - 1) {
                  curr_stream << VERT_BAR;
               }
            }
            // extend the current line i by the new information
            line_streams[i] << curr_stream.str();
         }
      }
      for(auto &stream : line_streams) {
         board_print << stream.str() << VERT_BAR << "\n";
      }
      board_print << init_space << VERT_BAR << h_border << VERT_BAR << "\n";
   }
   // column width for the row index plus vertical dash
   board_print << std::string(static_cast< unsigned long >(row_ind_space), ' ');
   // print the column index rows
   for(int i = 0; i < dim_x; ++i) {
      board_print << aze::utils::center(std::to_string(i), H_SIZE_PER_PIECE + 1, " ");
   }
   board_print << "\n";
   return board_print.str();
}
template <>
std::string enum_name(Status e)
{
   constexpr aze::utils::CEMap< Status, std::string_view, 4 > name_lookup = {
      std::pair{Status::TIE, std::string_view("TIE")},
      std::pair{Status::WIN_RED, std::string_view("WIN_RED")},
      std::pair{Status::WIN_BLUE, std::string_view("WIN_BLUE")},
      std::pair{Status::ONGOING, std::string_view("ONGOING")},
   };
   return std::string(name_lookup.at(e));
}
template <>
std::string enum_name(Team e)
{
   constexpr aze::utils::CEMap< Team, std::string_view, 3 > name_lookup = {
      std::pair{Team::BLUE, std::string_view("BLUE")},
      std::pair{Team::RED, std::string_view("RED")},
      std::pair{Team::NEUTRAL, std::string_view("NEUTRAL")},
   };
   return std::string(name_lookup.at(e));
}
template <>
std::string enum_name(FightOutcome e)
{
   constexpr aze::utils::CEMap< FightOutcome, std::string_view, 3 > name_lookup = {
      std::pair{FightOutcome::kill, std::string_view("kill")},
      std::pair{FightOutcome::death, std::string_view("death")},
      std::pair{FightOutcome::tie, std::string_view("tie")},
   };
   return std::string(name_lookup.at(e));
}
template <>
std::string enum_name(Token e)
{
   constexpr aze::utils::CEMap< Token, std::string_view, 13 > name_lookup = {
      std::pair{Token::flag, std::string_view("Flag")},
      std::pair{Token::spy, std::string_view("spy")},
      std::pair{Token::scout, std::string_view("scout")},
      std::pair{Token::miner, std::string_view("miner")},
      std::pair{Token::sergeant, std::string_view("sergeant")},
      std::pair{Token::lieutenant, std::string_view("lieutenant")},
      std::pair{Token::captain, std::string_view("captain")},
      std::pair{Token::major, std::string_view("major")},
      std::pair{Token::colonel, std::string_view("colonel")},
      std::pair{Token::general, std::string_view("general")},
      std::pair{Token::marshall, std::string_view("marshall")},
      std::pair{Token::bomb, std::string_view("bomb")},
      std::pair{Token::hole, std::string_view("hole")},
   };
   return std::string(name_lookup.at(e));
}

}  // namespace stratego::utils