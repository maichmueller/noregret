#include "utils.hpp"

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

   // piece string lambda function that returns a str of the kin
   // "-1 \n
   // 10.1 \n
   //   1"
   auto create_piece_str = [&H_SIZE_PER_PIECE, &mid, &team, &hide_unknowns](
                              const std::optional< Piece > &piece_opt, int line) {
      if(not piece_opt.has_value()) {
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
      }
      const auto &piece = piece_opt.value();
      std::string color = BLUE;  // blue by default
      if(piece.token() == Token::hole)
         // piece is an hole -> return a gray colored field
         return "\x1B[30;47m" + aze::utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
      else if(piece.team() == Team::RED) {
         color = RED;  // background red, text "white"
      }
      if(line == mid - 1) {
         // hidden info line
         std::string h = piece.flag_hidden() ? "?" : " ";
         return color + aze::utils::center(h, H_SIZE_PER_PIECE, " ") + RESET;
      } else if(line == mid) {
         // type and version info line
         if(hide_unknowns && piece.flag_hidden()) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + RESET;
         }
         const auto &token = piece.token();
         return color
                + aze::utils::center(
                   std::to_string(static_cast< int >(token)), H_SIZE_PER_PIECE, " ")
                + RESET;
      } else if(line == mid + 1) {
#if defined(_MSC_VER)
         // team info line
         return aze::utils::center(
            piece.team() == aze::Team::BLUE ? "B" : "R", H_SIZE_PER_PIECE, " ");
#else
         // for non msvc we have colored boxed to work for us
         return color + aze::utils::center("", H_SIZE_PER_PIECE, " ") + RESET;
#endif
      } else
         // empty line
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
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
   for(int row = dim_y - 1; row > -1; --row) {
      // per piece we have V_SIZE_PER_PIECE many lines to fill consecutively.
      // Iterate over every column and append the new segment to the right line.
      std::vector< std::stringstream > line_streams(static_cast< unsigned int >(V_SIZE_PER_PIECE));

      for(int col = 0; col < dim_x; ++col) {
         if(team == aze::Team::RED) {
            curr_piece = board(dim_x - 1 - row, dim_y - 1 - col);
         } else
            curr_piece = board(row, col);

         for(int i = 0; i < V_SIZE_PER_PIECE; ++i) {
            std::stringstream curr_stream;

            if(i == mid - 1 || i == mid + 1) {
               if(col == 0) {
                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space), ' ');
               }
               curr_stream << VERT_BAR << create_piece_str(curr_piece, i);
            } else if(i == mid) {
               if(col == 0) {
                  if(row < 10)
                     curr_stream << " " << row;
                  else
                     curr_stream << row;

                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space - 2), ' ')
                              << VERT_BAR;
               }
               curr_stream << create_piece_str(curr_piece, i);
               if(col != dim_x - 1)
                  curr_stream << VERT_BAR;
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

}  // namespace stratego::utils