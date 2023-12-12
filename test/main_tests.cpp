#include <gtest/gtest.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

int main(int argc, char **argv)
{
   spdlog::set_level(spdlog::level::level_enum{SPDLOG_ACTIVE_LEVEL});
   testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
