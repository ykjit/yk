void __llvm_deoptimize() { printf("deopt\n"); }

void foo() {
  int x = 11;
  int y = 13;
  int z;
  if (x == 11) {
    z = 3;
  } else {
    // Insert deopt here.
    z = 4;
  }
  printf("%d %d %d", x, y, z);
}

int main() { foo(); }
