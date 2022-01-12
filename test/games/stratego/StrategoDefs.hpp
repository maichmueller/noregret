#pragma once

namespace aze {

enum class Status {
   ONGOING = 404,
   TIE = 0,
   WIN_BLUE = 1,
   WIN_RED = -1,
};
enum class Team { BLUE = 0, RED = 1 };

}  // namespace aze