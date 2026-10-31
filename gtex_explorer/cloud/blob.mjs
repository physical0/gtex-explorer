// File bodies stay in local streams. Never print storage credentials or data.
import { get, head, put, BlobPreconditionFailedError, BlobNotFoundError } from '@vercel/blob';
import { createReadStream, createWriteStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';
const [operation, key, filename, encoded] = process.argv.slice(2);
const options = JSON.parse(encoded ?? '{}');
try {
  if (operation === 'get') {
    const result = await get(key, {access: 'private', useCache: false});
    if (!result) {
      process.stdout.write(JSON.stringify({found: false}));
    } else {
      await pipeline(result.stream, createWriteStream(filename));
      let etag = result.blob.etag;
      // Compressed HTTP representations can expose W/"...". Conditional writes
      // require the strong object ETag, and it must still describe these bytes.
      if (key.endsWith('/state.json') && etag.startsWith('W/')) {
        const metadata = await head(key);
        if (metadata.etag !== etag.slice(2)) throw new BlobPreconditionFailedError();
        etag = metadata.etag;
      }
      process.stdout.write(JSON.stringify({found: true, etag}));
    }
  } else if (operation === 'put') {
    const result = await put(key, createReadStream(filename), {
      access: 'private', addRandomSuffix: false,
      allowOverwrite: !options.create,
      ...(options.expected ? {ifMatch: options.expected} : {}),
    });
    process.stdout.write(JSON.stringify({etag: result.etag}));
  } else {
    throw new Error('Unknown storage operation');
  }
} catch (error) {
  if (error instanceof BlobNotFoundError && operation === 'get') {
    process.stdout.write(JSON.stringify({found: false}));
  } else {
    // Existing immutable objects and CAS conflicts both use exit code 3.
    process.stderr.write(error?.name ?? 'BlobError');
    process.exitCode = error instanceof BlobPreconditionFailedError || /already exists/i.test(error?.message ?? '') ? 3 : 1;
  }
}
