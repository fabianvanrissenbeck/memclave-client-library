/** vci-msg.h - header defining the message format used to communicate with virtual CIs */

#ifndef COMMON_VCI_H
#define COMMON_VCI_H

#ifndef COMMON_VCI_PREVENT_INCLUDES
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#endif

#define VCI_SOCKET_NAME "/tmp/vci.sock"

typedef enum vci_msg_type {
    /** returned when receiving invalid message */
    VCI_MSG_ERR = -3,
    /** local system error */
    VCI_SYS_ERR = -2,
    /** returned if DPUs are in an inconsistent state */
    VCI_ERR = -1,
    /** standard response for most commands */
    VCI_OK,
    /** check whether a given rank is present */
    VCI_PRESENT,
    /** response to VCI_PRESENT */
    VCI_IS_PRESENT,
    /** get the current state of the DPU MUX */
    VCI_QRY_MUX,
    /** response to VCI_QRY_MUX */
    VCI_QRY_RES,
    /** release control over the MUX and resume computation on the DPU */
    VCI_REL_MUX,
} vci_msg_type;

/** message structure passed over the unix socket */
typedef struct vci_msg {
    int8_t type;
    uint8_t n_faulted;
    uint8_t n_running;
    uint8_t _pad;
    uint32_t rank_nr;
} vci_msg;

_Static_assert(sizeof(vci_msg) == 8, "ci_msg struct packed incorrectly");
_Static_assert(offsetof(vci_msg, n_faulted) == 1, "ci_msg incorrectly ordered");
_Static_assert(offsetof(vci_msg, n_running) == 2, "ci_msg incorrectly ordered");
_Static_assert(offsetof(vci_msg, _pad) == 3, "ci_msg incorrectly ordered");
_Static_assert(offsetof(vci_msg, rank_nr) == 4, "ci_msg incorrectly ordered");

static inline uint64_t vci_msg_to_qword(vci_msg msg) {
    uint64_t res;

    _Static_assert(sizeof(msg) == sizeof(res), "vci_msg packed incorrectly");
    memcpy(&res, &msg, sizeof(msg));

    return res;
}

static inline vci_msg vci_msg_from_qword(uint64_t word) {
    vci_msg res;

    _Static_assert(sizeof(word) == sizeof(res), "vci_msg packed incorrectly");
    memcpy(&res, &word, sizeof(word));

    return res;
}

static inline void vci_msg_to_string(vci_msg msg, char buf[120]) {
    static const char* type_to_string[] = {
        "VCI_MSG_ERR",
        "VCI_SYS_ERR",
        "VCI_ERR",
        "VCI_OK",
        "VCI_PRESENT",
        "VCI_IS_PRESENT",
        "VCI_QRY_MUX",
        "VCI_QRY_RES",
        "VCI_REL_MUX",
    };

    if (msg.type == VCI_QRY_RES) {
        snprintf(
            buf, 120,
            "{ .type = %s, .n_faulted = %u, .n_done = %u, .rank_nr = %02u }",
            type_to_string[msg.type - VCI_MSG_ERR], msg.n_faulted, msg.n_running, msg.rank_nr
        );
    } else if (msg.type >= VCI_MSG_ERR && msg.type <= VCI_REL_MUX) {
        snprintf(
            buf, 120,
            "{ .type = %s, .rank_nr = %02u }", type_to_string[msg.type - VCI_MSG_ERR], msg.rank_nr
        );
    } else {
        snprintf(
            buf, 120,
            "{ .type = %02x, .rank_nr = %02u }", (uint8_t) msg.type, msg.rank_nr
        );
    }
}

#endif
