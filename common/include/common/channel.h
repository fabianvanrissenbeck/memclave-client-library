/** channel.h - module used to create a channel between client and router supporting large data transfers via shared memory */

#ifndef COMMON_CHANNEL_H
#define COMMON_CHANNEL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#define COMMON_GNU_SOURCE_DEFINED
#endif

#include "common/dpu-msg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <poll.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>

#define ROUTER_SOCK_NAME "/tmp/vpim-router.sock"
#define CLIENT_SOCK_NAME "/tmp/vpim-client.sock"

typedef struct channel {
    int sock;
    struct sockaddr_un conn_addr;
} channel;

static channel* channel_init(const char* self_name, const char* conn_name) {
    size_t self_len = strlen(self_name);
    size_t conn_len = strlen(conn_name);

    if (self_len >= sizeof(((struct sockaddr_un*) 0)->sun_path)) {
        puts("self_len is to long");
        return NULL;
    }

    if (conn_len >= sizeof(((struct sockaddr_un*) 0)->sun_path)) {
        puts("self_len is to long");
        return NULL;
    }

    struct sockaddr_un self_addr;
    struct sockaddr_un conn_addr;

    self_addr.sun_family = AF_UNIX;
    conn_addr.sun_family = AF_UNIX;

    memcpy(self_addr.sun_path, self_name, self_len);
    memcpy(conn_addr.sun_path, conn_name, conn_len);

    self_addr.sun_path[self_len] = '\0';
    conn_addr.sun_path[conn_len] = '\0';

    int sock = socket(AF_UNIX, SOCK_DGRAM, 0);

    if (sock < 0) {
        perror("cannot create socket");
        return NULL;
    }

    unlink(self_name);

    if (bind(sock, (const struct sockaddr*) &self_addr, sizeof(self_addr)) < 0) {
        perror("cannot bind to address");

        close(sock);
        return NULL;
    }

    channel* res = malloc(sizeof(*res));

    if (res == NULL) {
        perror("malloc failure");

        close(sock);
        return NULL;
    }

    res->sock = sock;
    res->conn_addr = conn_addr;

    return res;
}

static void channel_free(channel* chan) {
    if (chan != NULL && chan->sock >= 0) {
        close(chan->sock);
    }

    free(chan);
}

static bool channel_can_receive(channel* chan) {
    struct pollfd fd = {
        .fd = chan->sock,
        .events = POLLIN
    };

    if (poll(&fd, 1, 0) < 0) {
        perror("cannot poll");
        return false;
    }

    return (fd.revents & POLLIN) != 0;
}

/**
 * @brief transmit data over the channel including a buffer referred to by fd
 * @param hdr message header to transmit
 * @param fd file descriptor send as control message - will be closed when returning
 * @returns 0 on success or a negative value on failure
 */
static int channel_send_fd(channel* chan, msg_header hdr, int fd) {
    struct msghdr msg_hdr = {
        .msg_name = &chan->conn_addr,
        .msg_namelen = sizeof(chan->conn_addr),
        .msg_iov = &(struct iovec) {
            .iov_base = &hdr,
            .iov_len = sizeof(hdr)
        },
        .msg_iovlen = 1,
    };

    uint8_t cmsg_buf[CMSG_SPACE(sizeof(fd))];
    struct cmsghdr* cmsg_hdr = (struct cmsghdr*) &cmsg_buf[0];

    if (fd >= 0) {
        cmsg_hdr->cmsg_level = SOL_SOCKET;
        cmsg_hdr->cmsg_type = SCM_RIGHTS;
        cmsg_hdr->cmsg_len = CMSG_LEN(sizeof(fd));

        memcpy(CMSG_DATA(cmsg_hdr), &fd, sizeof(fd));

        msg_hdr.msg_control = cmsg_hdr;
        msg_hdr.msg_controllen = CMSG_SPACE(sizeof(fd));
    }

    if (sendmsg(chan->sock, &msg_hdr, 0) < 0) {
        perror("cannot send message");

        if (fd >= 0) {
            close(fd);
        }

        return -1;
    }

    if (fd >= 0) {
        close(fd);
    }

    return 0;
}

/**
 * @brief transmit data over the channel which may include a large buffer
 * @param hdr message header to transmit
 * @param n Size of the optional buffer or zero if no buffer should be sent. Multiple of 8.
 * @param buf optional buffer with a capacity of at least n bytes
 * @returns 0 on success or a negative value on failure
 */
static int channel_send(channel* chan, msg_header hdr, size_t n, const void* buf) {
    if (n % 8 != 0) {
        puts("n must be a multiple of 8");
        return -1;
    }

    int mem_fd = -1;

    if (n > 0) {
        mem_fd = memfd_create("channel", 0);

        if (mem_fd < 0) {
            perror("cannot create shared anonymous file");
            return -1;
        }

        if (ftruncate(mem_fd, (off_t) n) < 0) {
            perror("cannot reserve shared memory");

            close(mem_fd);
            return -1;
        }

        void* shm = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);

        if (shm == MAP_FAILED) {
            perror("cannot map shared memory");

            close(mem_fd);
            return -1;
        }

        memcpy(shm, buf, n);
        munmap(shm, n);
    }

    return channel_send_fd(chan, hdr, mem_fd);
}

static int channel_recv_fd(channel* chan, msg_header* out_hdr, int* out_fd) {
    struct msghdr msg_hdr = {
        .msg_name = &chan->conn_addr,
        .msg_namelen = sizeof(chan->conn_addr),
        .msg_iov = &(struct iovec) {
            .iov_base = out_hdr,
            .iov_len = sizeof(*out_hdr)
        },
        .msg_iovlen = 1,
    };

    int mem_fd = -1;
    uint8_t cmsg_buf[CMSG_SPACE(sizeof(mem_fd))];

    struct cmsghdr* cmsg_hdr = (struct cmsghdr*) cmsg_buf;

    cmsg_hdr->cmsg_level = SOL_SOCKET;
    cmsg_hdr->cmsg_type = SCM_RIGHTS;
    cmsg_hdr->cmsg_len = CMSG_LEN(sizeof(mem_fd));

    if (out_fd != NULL) {
        msg_hdr.msg_control = cmsg_hdr;
        msg_hdr.msg_controllen = CMSG_SPACE(sizeof(mem_fd));
    }

    if (recvmsg(chan->sock, &msg_hdr, 0) < 0) {
        perror("cannot receive message");
        return -1;
    }

    if (msg_hdr.msg_controllen > 0) {
        memcpy(out_fd, CMSG_DATA(cmsg_hdr), sizeof(*out_fd));
    } else {
        if (out_fd) {
            *out_fd = -1;
        }
    }

    return 0;
}

/**
 * @brief receive data over the channel which may include a large buffer
 * @param out_hdr destination to write received header to
 * @param buf destination to write pointer to optional buffer to or NULL
 * @param out_n destination to write size of received optional buffer to or NULL
 * @returns 0 on success or a negative value on failure
 */
static int channel_recv(channel* chan, msg_header *out_hdr, void** out_buf, size_t* out_n) {
    int mem_fd = -1;
    int res;

    if (out_buf) {
        res = channel_recv_fd(chan, out_hdr, &mem_fd);
    } else {
        res = channel_recv_fd(chan, out_hdr, NULL);
    }

    if (res < 0) {
        return res;
    }

    if (mem_fd > 0) {
        struct stat stat_buf;

        if (fstat(mem_fd, &stat_buf) < 0) {
            perror("cannot stat shared memory object");

            close(mem_fd);
            return -1;
        }

        size_t mem_sz = stat_buf.st_size;
        void* buf = mmap(NULL, mem_sz, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);

        if (buf == MAP_FAILED) {
            perror("cannot map shared memory");

            close(mem_fd);
            return -1;
        }

        close(mem_fd);

        if (out_buf && out_n) {
            *out_buf = buf;
            *out_n = mem_sz;
        } else {
            munmap(buf, mem_sz);
        }
    } else {
        if (out_buf) {
            *out_buf = NULL;
        }

        if (out_n) {
            *out_n = 0;
        }
    }

    return 0;
}

#ifdef COMMON_GNU_SOURCE_DEFINED
#undef _GNU_SOURCE
#undef COMMON_GNU_SOURCE_DEFINED
#endif

#endif
