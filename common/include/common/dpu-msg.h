#ifndef COMMON_DPU_MSG_H
#define COMMON_DPU_MSG_H

#include "aes.h"

#include <stddef.h>
#include <stdbool.h>

typedef enum msg_type {
    MSG_NOP = 0x0,

    // begin unauthenticated client issued messages
    MSG_SET_KEY,
    MSG_CHANGE_KEY,
    // begin authenticated client issued messages
    MSG_PING = 0x10,

    MSG_COPY_DECRYPT,
    MSG_COPY_ENCRYPT,
    MSG_BEGIN_CLASSIFY,
    MSG_BEGIN_MULTIPLY,
    MSG_BEGIN_SUM,

    // begin client to router issued messages
    MSG_ALLOC_RANKS = 0x30,
    MSG_DEALLOC_RANKS,
    MSG_FLUSH_QUEUE,

    // begin router to client issued messages
    MSG_ALLOCATED,

    // begin DPU issued messages
    MSG_ACK = 0x40,
    MSG_INVALID,
    MSG_PONG,
    MSG_ACK_COPY,
} msg_type;

enum {
    MSG_ID_ROUTER = 0xFFD,
    MSG_ID_CLIENT = 0xFFE,
    MSG_ID_BROADCAST = 0xFFF
};

static inline uint16_t msg_get_id_for(uint8_t rank, uint8_t dpu) {
    return (uint16_t) rank << 6 | (dpu & 0x3F);
}

static inline uint8_t msg_rank_from_id(uint16_t id) {
    return id >> 6;
}

static inline uint8_t msg_dpu_from_id(uint16_t id) {
    return id & 0x3F;
}

/**
 * [0-6]    Message Type
 * [7]      Broadcast Flag
 * [8-19]   Sender ID
 * [20-31]  Receiver ID
 */
typedef struct msg_header_param {
    uint32_t type : 8,
             sender_id : 12,
             receiver_id : 12;
} msg_header_param;

static inline msg_type msg_get_type(msg_header_param p) {
    return p.type;
}

static inline uint16_t msg_get_sender_id(msg_header_param p) {
    return p.sender_id;
}

static inline uint16_t msg_get_receiver_id(msg_header_param p) {
    return p.receiver_id;
}

/**
 * Header of all messages transmitted between DPU and host. This header is usually
 * authenticated. Unused fields must be set to zero.
 */
typedef struct msg_header {
    msg_header_param param;
    /** message count used while encrypting the message */
    uint32_t count;
    /** MAC of the msg_header structure, set to zero when calculating the MAC. */
    uint64_t mac;
    /** optional data which is encrypted (via CTR- or CCM-mode) */
    union {
        /** included in some messages of the MSG_ACK specifiying the amount of cycles processing took */
        uint64_t cycles;
        uint64_t ping;
        aes_block otp;
        aes_block key;
        struct {
            aes_block key;
            aes_block nonce;
            aes_block otp;
        } change_key;
        /**
         * If type is COPY_ENCRYPT, encrypt data in target and move it to g_msg_data.
         * If type is COPY_DECRYPT, decrypt data in g_msg_data and move it to target.
         * The data_mac is computed independently of the other mac. This way we can avoid
         * unauthorized data transfers.
         */
        struct {
            uint32_t size;
            uint32_t target;
            uint32_t count;
            uint64_t data_mac;
        } copy;
        struct {
            uint32_t model_addr;
            uint32_t data_addr;
            uint32_t data_sz;
            uint32_t output_addr;
        } classify;
        struct {
            uint32_t lhs_addr;
            uint32_t rhs_addr;
            uint16_t lhs_rows;
            union {
                uint16_t lhs_cols;
                uint16_t rhs_rows;
            };
            uint16_t rhs_cols;
            uint32_t out_addr;
        } multiply;
        struct {
            uint32_t lhs_addr;
            uint32_t rhs_addr;
            uint32_t size;
            uint32_t out_addr;
            /** 1 if the input values are IEEE-754 floats or 0 if they are signed integers */
            uint32_t sum_floats;
        } sum;
        /** payload for router control messages */
        struct {
            union {
                /** for allocation */
                unsigned nr_ranks;
                /** for deallocation, flushing, alloc acknowledgement */
                uint64_t map;
            };
        } control;
    } opt;
} msg_header;

_Static_assert(sizeof(msg_header) == 64, "size of msg_header must be the same as in the linker script");
_Static_assert(_Alignof(msg_header) % 8 == 0, "msg_header must be 8-byte aligned");
_Static_assert(_Alignof(((msg_header*) 0)->opt) % 8 == 0, "optional data must be 8-byte aligned for decryption");

#endif
