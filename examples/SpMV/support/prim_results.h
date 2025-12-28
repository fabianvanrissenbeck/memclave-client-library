#ifndef PRIM_RESULTS_H
#define PRIM_RESULTS_H

// Header-only CSV "upsert" for PRIM/Memclave benchmarks.
// - Keyed by first column "Test"
// - Updates only the column you pass (e.g., "CPU", "DPU", "M_C2D", ...)
// - Creates file with header if missing
// - Adds row if test not present
// - Preserves other columns/fields
// - Atomic rewrite (tmp + rename)
//
// Usage:
//   update_csv_from_timer("results.csv", "TRNS", &timer, 0, p.n_reps, "CPU");
//   update_csv_from_timer("results.csv", "TRNS", &timer, 1, p.n_reps, "DPU");
//
// Or if DPU is sum of two timers:
//   double dpu_ms = prim_timer_ms_avg(&timer, k0, reps) + prim_timer_ms_avg(&timer, k1, reps);
//   update_csv("results.csv", "TRNS", "DPU", dpu_ms);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#if defined(__linux__)
#include <unistd.h>
#endif

// #define PRIM_RESULTS_USE_FLOCK 1
#if defined(PRIM_RESULTS_USE_FLOCK)
#include <sys/file.h>
#endif

// Forward declare Timer if you don't want to include your timer header here.
// But easiest is: include this AFTER support/timer.h in your host file.
typedef struct Timer Timer;

// ------------------------ Configuration ------------------------

static const char *const PRIM_RESULTS_REQUIRED_COLS[] = {
    "Test", "CPU", "DPU", "M_C2D", "M_D2C", "UPMEM", "U_C2D", "U_D2C"
};
enum { PRIM_RESULTS_REQUIRED_NCOLS = 8 };

// Format used when writing numeric values to CSV
#ifndef PRIM_RESULTS_VALUE_FMT
#define PRIM_RESULTS_VALUE_FMT "%.3f"
#endif

// ------------------------ Timer helpers ------------------------
#if 0
static inline double prim_timer_ms_avg(const Timer *timer, int i, int reps) {
    // Matches your print(): timer->time[] is in microseconds accumulated.
    // Avg ms = us / (1000 * REP)
    if (reps <= 0) reps = 1;
    // We cannot access Timer layout here unless timer.h is included before this header.
    // So this function will compile only if Timer has "time" as in PRIM.
    return ((const double *)timer->time)[i] / (1000.0 * (double)reps);
}

static inline double prim_timer_ms_avg_sum(const Timer *timer, const int *idxs, int n, int reps) {
    double s = 0.0;
    for (int k = 0; k < n; k++) s += prim_timer_ms_avg(timer, idxs[k], reps);
    return s;
}
#endif

// ------------------------ Small CSV utilities ------------------------

static inline int prim__needs_csv_quote(const char *s) {
    for (const char *p = s; *p; p++) {
        if (*p == ',' || *p == '"' || *p == '\n' || *p == '\r') return 1;
    }
    return 0;
}

static inline void prim__csv_write_cell(FILE *f, const char *s) {
    if (!s) s = "";
    if (!prim__needs_csv_quote(s)) {
        fputs(s, f);
        return;
    }
    fputc('"', f);
    for (const char *p = s; *p; p++) {
        if (*p == '"') fputc('"', f); // escape quote by doubling
        fputc(*p, f);
    }
    fputc('"', f);
}

// Split a CSV line into cells (supports basic quoting with double quotes).
// Returns malloc'd array of malloc'd strings. out_n set to count.
static inline char **prim__csv_split_line(const char *line, int *out_n) {
    int cap = 16, n = 0;
    char **cells = (char **)calloc((size_t)cap, sizeof(char *));
    if (!cells) return NULL;

    const char *p = line;
    while (*p && (*p == '\n' || *p == '\r')) p++;

    while (*p) {
        if (n >= cap) {
            cap *= 2;
            char **tmp = (char **)realloc(cells, (size_t)cap * sizeof(char *));
            if (!tmp) { free(cells); return NULL; }
            cells = tmp;
        }

        // Parse one cell
        int in_quote = 0;
        size_t bufcap = 64, buflen = 0;
        char *buf = (char *)malloc(bufcap);
        if (!buf) { free(cells); return NULL; }

        if (*p == '"') { in_quote = 1; p++; }

        while (*p) {
            if (in_quote) {
                if (*p == '"') {
                    if (*(p + 1) == '"') { // escaped quote
                        if (buflen + 1 >= bufcap) { bufcap *= 2; buf = (char *)realloc(buf, bufcap); }
                        buf[buflen++] = '"';
                        p += 2;
                        continue;
                    } else {
                        p++; // end quote
                        in_quote = 0;
                        continue;
                    }
                }
            } else {
                if (*p == ',') { p++; break; }
                if (*p == '\n' || *p == '\r') { break; }
            }

            if (buflen + 1 >= bufcap) { bufcap *= 2; buf = (char *)realloc(buf, bufcap); }
            buf[buflen++] = *p++;
        }

        buf[buflen] = '\0';
        cells[n++] = buf;

        // consume line ending
        while (*p && (*p == '\r' || *p == '\n')) p++;
        // if not at comma, and not at end, continue naturally
    }

    *out_n = n;
    return cells;
}

static inline void prim__csv_free_cells(char **cells, int n) {
    if (!cells) return;
    for (int i = 0; i < n; i++) free(cells[i]);
    free(cells);
}

static inline int prim__col_index(char **header, int ncols, const char *name) {
    for (int i = 0; i < ncols; i++) {
        if (header[i] && strcmp(header[i], name) == 0) return i;
    }
    return -1;
}

// Ensure required columns exist; append missing ones to header and all rows.
static inline int prim__ensure_required_cols(
    char ***p_header, int *p_ncols,
    char ****p_rows, int *p_nrows
) {
    char **header = *p_header;
    int ncols = *p_ncols;

    for (int rc = 0; rc < PRIM_RESULTS_REQUIRED_NCOLS; rc++) {
        const char *need = PRIM_RESULTS_REQUIRED_COLS[rc];
        if (prim__col_index(header, ncols, need) >= 0) continue;

        // append column
        char **new_header = (char **)realloc(header, (size_t)(ncols + 1) * sizeof(char *));
        if (!new_header) return -1;
        header = new_header;
        header[ncols] = strdup(need);
        if (!header[ncols]) return -1;

        // extend each row with empty cell
        for (int r = 0; r < *p_nrows; r++) {
            char **row = (*p_rows)[r];
            char **new_row = (char **)realloc(row, (size_t)(ncols + 1) * sizeof(char *));
            if (!new_row) return -1;
            (*p_rows)[r] = new_row;
            (*p_rows)[r][ncols] = strdup("");
            if (!(*p_rows)[r][ncols]) return -1;
        }

        ncols++;
    }

    *p_header = header;
    *p_ncols = ncols;
    return 0;
}

// ------------------------ Core API ------------------------

// Upsert a single numeric metric into the CSV table.
static inline int update_csv(
    const char *csv_path,
    const char *test_name,
    const char *metric_name, // one of: CPU, DPU, M_C2D, M_D2C, UPMEM, U_C2D, U_D2C (or your custom col)
    double value_ms
) {
    if (!csv_path || !test_name || !metric_name) return -1;

    FILE *in = fopen(csv_path, "r");
#if defined(PRIM_RESULTS_USE_FLOCK)
    if (in) flock(fileno(in), LOCK_EX);
#endif

    char **header = NULL;
    int ncols = 0;

    char ***rows = NULL;
    int nrows = 0;
    int rows_cap = 0;

    if (!in) {
        // File does not exist yet: create with required header.
        ncols = PRIM_RESULTS_REQUIRED_NCOLS;
        header = (char **)calloc((size_t)ncols, sizeof(char *));
        if (!header) return -1;
        for (int i = 0; i < ncols; i++) header[i] = strdup(PRIM_RESULTS_REQUIRED_COLS[i]);
    } else {
        // Read header line
        char *line = NULL;
        size_t len = 0;
        ssize_t r = getline(&line, &len, in);
        if (r <= 0) { free(line); fclose(in); return -1; }

        header = prim__csv_split_line(line, &ncols);
        free(line);
        if (!header) { fclose(in); return -1; }

        // Read rows
        while (1) {
            line = NULL; len = 0;
            r = getline(&line, &len, in);
            if (r <= 0) { free(line); break; }
            int cn = 0;
            char **cells = prim__csv_split_line(line, &cn);
            free(line);
            if (!cells) { fclose(in); return -1; }

            // Normalize row width to ncols (pad with empty)
            if (cn < ncols) {
                char **tmp = (char **)realloc(cells, (size_t)ncols * sizeof(char *));
                if (!tmp) { prim__csv_free_cells(cells, cn); fclose(in); return -1; }
                cells = tmp;
                for (int i = cn; i < ncols; i++) cells[i] = strdup("");
                cn = ncols;
            } else if (cn > ncols) {
                // If row is wider than header, extend header with generic names
                for (int i = ncols; i < cn; i++) {
                    char colname[32];
                    snprintf(colname, sizeof(colname), "col_%d", i);
                    char **new_header = (char **)realloc(header, (size_t)(i + 1) * sizeof(char *));
                    if (!new_header) { prim__csv_free_cells(cells, cn); fclose(in); return -1; }
                    header = new_header;
                    header[i] = strdup(colname);
                }
                ncols = cn;
            }

            if (nrows >= rows_cap) {
                rows_cap = rows_cap ? rows_cap * 2 : 16;
                char ***tmp = (char ***)realloc(rows, (size_t)rows_cap * sizeof(char **));
                if (!tmp) { prim__csv_free_cells(cells, cn); fclose(in); return -1; }
                rows = tmp;
            }
            rows[nrows++] = cells;
        }

        fclose(in);
    }

    // Ensure required cols exist
    if (prim__ensure_required_cols(&header, &ncols, &rows, &nrows) != 0) return -1;

    // Ensure the metric column exists (allow custom columns too)
    int col = prim__col_index(header, ncols, metric_name);
    if (col < 0) {
        // append metric column
        char **new_header = (char **)realloc(header, (size_t)(ncols + 1) * sizeof(char *));
        if (!new_header) return -1;
        header = new_header;
        header[ncols] = strdup(metric_name);
        if (!header[ncols]) return -1;

        for (int r = 0; r < nrows; r++) {
            char **new_row = (char **)realloc(rows[r], (size_t)(ncols + 1) * sizeof(char *));
            if (!new_row) return -1;
            rows[r] = new_row;
            rows[r][ncols] = strdup("");
            if (!rows[r][ncols]) return -1;
        }
        col = ncols;
        ncols++;
    }

    // Find (or create) the test row by "Test" column
    int test_col = prim__col_index(header, ncols, "Test");
    if (test_col < 0) test_col = 0;

    int row_idx = -1;
    for (int r = 0; r < nrows; r++) {
        if (rows[r][test_col] && strcmp(rows[r][test_col], test_name) == 0) {
            row_idx = r;
            break;
        }
    }
    if (row_idx < 0) {
        // append new row
        char **new_row = (char **)calloc((size_t)ncols, sizeof(char *));
        if (!new_row) return -1;
        for (int c = 0; c < ncols; c++) new_row[c] = strdup("");
        free(new_row[test_col]);
        new_row[test_col] = strdup(test_name);

        if (nrows >= rows_cap) {
            rows_cap = rows_cap ? rows_cap * 2 : 16;
            char ***tmp = (char ***)realloc(rows, (size_t)rows_cap * sizeof(char **));
            if (!tmp) return -1;
            rows = tmp;
        }
        rows[nrows++] = new_row;
        row_idx = nrows - 1;
    }

    // Update only the requested metric cell
    char buf[64];
    snprintf(buf, sizeof(buf), PRIM_RESULTS_VALUE_FMT, value_ms);

    free(rows[row_idx][col]);
    rows[row_idx][col] = strdup(buf);
    if (!rows[row_idx][col]) return -1;

    // Write atomically
    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", csv_path);

    FILE *out = fopen(tmp_path, "w");
    if (!out) return -1;

    // header
    for (int c = 0; c < ncols; c++) {
        if (c) fputc(',', out);
        prim__csv_write_cell(out, header[c]);
    }
    fputc('\n', out);

    // rows
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            if (c) fputc(',', out);
            prim__csv_write_cell(out, rows[r][c]);
        }
        fputc('\n', out);
    }

    fclose(out);

#if defined(__linux__)
    // rename is atomic on POSIX when same filesystem
    if (rename(tmp_path, csv_path) != 0) return -1;
#else
    // fallback: best-effort
    remove(csv_path);
    if (rename(tmp_path, csv_path) != 0) return -1;
#endif

    // cleanup
    for (int c = 0; c < ncols; c++) free(header[c]);
    free(header);
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) free(rows[r][c]);
        free(rows[r]);
    }
    free(rows);

    return 0;
}

#if 0
// compute avg ms from Timer slot and write to CSV.
static inline int update_csv_from_timer(
    const char *csv_path,
    const char *test_name,
    const Timer *timer,
    int timer_idx,
    int reps,
    const char *metric_name
) {
    double ms = prim_timer_ms_avg(timer, timer_idx, reps);
    return update_csv(csv_path, test_name, metric_name, ms);
}
#endif
#endif // PRIM_RESULTS_H
