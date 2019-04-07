/*
 * File   : loop.c
 * Author : Jiayuan Mao
 * Email  : maojiayuan@gmail.com
 * Date   : 04/06/2019
 *
 * Distributed under terms of the MIT license.
 */


long c_loop(long n) {
    long s;
    for (long i = 0; i < n; ++i) {
        s += i;
    }
    return s
}

