// Compiler:
// Run-time:

#include <yk.h>

int
main(int argc, char **argv)
{
    MT *mt = yk_mt();
    Location loc = yk_new_location();
    for (int i = 0; i < yk_mt_hot_threshold(mt); i++) {
        yk_control_point(mt, &loc);
    }
    yk_drop_location(&loc);
    return 0;
}
