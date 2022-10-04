#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

int main(int argc, char **argv)  {
char on = '1', off = '0';
    const char *cpu1 = "/sys/devices/system/cpu/cpu1/online";
    int fd = 0;
    char str[10];
    fd = open(cpu1, O_RDWR);
    printf("id= %d\n", fd);
    if (fd < 0)
        exit(0);


close(fd);
return 0;
}
