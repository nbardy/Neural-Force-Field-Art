// The function bytes_to_unicode translates every possible byte (0..255) into a Unicode character representing it visually.
function bytesToUnicode() {
  const bs = [...Array(33).keys()].slice(33, 126 + 1).concat([...Array(161).keys()].slice(161, 172 + 1), [...Array(174).keys()].slice(174, 255 + 1));
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 2**8; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(2**8 + n);
      n += 1;
    }
  }
  const d = Object.fromEntries(bs.map((v, i) => [v, String.fromCharCode(cs[i])]));
  return d;
}

// The function getPairs returns all bigrams as a set of tuples from the input word.
function getPairs(word) {
  const pairs = new Set();
  let prevChar = word[0];
  for (let char of word.slice(1)) {
    pairs.add([prevChar, char]);
    prevChar = char;
  }
  return pairs;
}

class Encoder {
  constructor(encoder, bpeMerges) {
    // Initializing variables as per the Python code.
    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = Object.fromEntries(Object.entries(this.byteEncoder).map(([k, v]) => [v, k]));
    this.encoder = encoder;
    this.decoder = Object.fromEntries(Object.entries(this.encoder).map(([k, v]) => [v, k]));
    this.bpeRanks = Object.fromEntries(bpeMerges.map((v, i) => [v, i]));
    this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.cache = {};
  }

  // The method bpe uses bpeRanks to iteratively merge all possible BPE tokens up the tree.
  bpe(token) {
    if (token in this.cache) {
      return this.cache[token];
    }
    // Rest of the logic as per Python code.
    // ...

    // Cache the result and return
    this.cache[token] = word;
    return word;
  }

  encode(text) {
    // string goes in, list of integers comes out
    // Logic for encoding as per Python code.
    // ...
  }

  encodeAndShowWork(text) {
    // debugging function, same as encode but returns all intermediate work
    // Logic for encoding and showing work as per Python code.
    // ...
  }

  decode(bpeIdx) {
    // list of integers comes in, string comes out
    // Logic for decoding as per Python code.
    // ...
  }
}

// The function getFile downloads remoteFile to localFile if necessary.
function getFile(localFile, remoteFile) {
  if (!fs.existsSync(localFile)) {
    console.log(`downloading ${remoteFile} to ${localFile}`);
    axios.get(remoteFile).then((response) => {
      fs.writeFileSync(localFile, response.data);
    });
  }
}

// The function getEncoder returns an instance of the GPT BPE Encoder/Decoder and handles caching of "database" files.
async getEncoder() {
    // Define the home directory and cache directory for downloading files
    const homeDir = (typeof window === 'undefined') ? require('os').homedir() : 'home/user';
    const cacheDir = homeDir + '/.cache/mingpt';

    // Define the remote file URLs
    const encoderRemoteFile = 'https://openaipublic.blob.core.windows.net/gpt-2/models/774M/encoder.json';
    const bpeRemoteFile = 'https://openaipublic.blob.core.windows.net/gpt-2/models/774M/vocab.bpe';

    // Function to download file if it does not exist
    const getFile = async (localFile, remoteFile) => {
      if (!require('fs').existsSync(localFile)) {
        console.log(`downloading ${remoteFile} to ${localFile}`);
        const response = await require('axios').get(remoteFile, { responseType: 'arraybuffer' });
        require('fs').writeFileSync(localFile, response.data);
      }
    };

    // Check and download files if necessary
    await getFile(cacheDir + '/encoder.json', encoderRemoteFile);
    await getFile(cacheDir + '/vocab.bpe', bpeRemoteFile);

    // Load encoder.json and vocab.bpe
    const encoder = JSON.parse(require('fs').readFileSync(cacheDir + '/encoder.json', 'utf-8'));
    const bpeMerges = require('fs').readFileSync(cacheDir + '/vocab.bpe', 'utf-8').split('\n').map(x => x.split(' '));

    // Create and return an instance of the Encoder class
    return new Encoder(encoder, bpeMerges);
  }
}
