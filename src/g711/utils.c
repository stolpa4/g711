#include "utils.h"

#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stddef.h>


float g711_s16_to_float(short sample)
{
    static const float conv_coef = 1.F / 32768.F;
    return conv_coef * (float)sample;
}


short g711_float_to_s16(float sample)
{
    int sample_s16 = (int) (sample * 32768.F);

    /* clip to a proper interval */
    if (sample_s16 < -32768) sample_s16 = -32768;
    else if (sample_s16 > 32767) sample_s16 = 32767;

    return sample_s16;
}


bool g711_validate_path(const char file_path[])
{
	assert(file_path);
	return strlen(file_path);
}


FILE* g711_open_file_for_read(const char file_path[])
{
	assert(file_path);
	return fopen(file_path, "rb");
}


unsigned long g711_get_file_bts_num(FILE* file)
{
	assert(file);

	/* Flush the internal buffer to obtain a true position */
	fflush(file);

	/* Remember the position to return to it after seeking */
	long int old_pos = ftell(file);
	if (old_pos < 0L) return 0UL;

	/* Get the number of bytes in the file */
	int err = fseek(file, 0L, SEEK_END);
	if (err) return 0UL;


	long int size = ftell(file);
	unsigned long result = size > 0L ? (unsigned long)size : 0UL;

	/* Returning to the initial position */
	fseek(file, old_pos, SEEK_SET);

	return result;
}


char* g711_read_file(FILE* file, unsigned long bts_num)
{
	assert(file);

	char* buffer = malloc(bts_num);
	if (!buffer) return NULL;

	unsigned long bts_read = fread(buffer, 1, bts_num, file);

	if (bts_read != bts_num) {
		free(buffer);
		return NULL;
	}

	return buffer;
}


FILE* g711_open_file_for_write(const char file_path[])
{
	assert(file_path);
	return fopen(file_path, "wb");	
}


bool g711_write_file(FILE* file, const char* bts, unsigned long bts_num)
{
	assert(file);
	assert(bts);
	assert(bts_num > 0);

	unsigned long bts_written = fwrite(bts, 1, bts_num, file);

	return bts_written == bts_num;
}


void g711_close_file(FILE* file)
{
	assert(file);
	fclose(file);
}
