/**
 * AudioWorklet processor for playing raw 16-bit PCM audio.
 *
 * Runs on the dedicated audio worklet thread (separate from the main JS thread
 * and the browser's audio rendering thread), which prevents main-thread work
 * from causing audio glitches.
 *
 * Protocol:
 *   Main thread sends:  ArrayBuffer of Int16 samples (little-endian, 16 kHz mono)
 *   Processor outputs:  Float32 mono audio to the default output channel
 */
'use strict';

class PCMPlayerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Internal queue of Float32 chunks waiting to be played.
        // Each chunk is a Float32Array of samples already converted from Int16.
        this._queue = [];
        this._queued = 0;   // total samples currently buffered

        this.port.onmessage = (event) => {
            const int16 = new Int16Array(event.data);
            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) {
                float32[i] = int16[i] / 32768.0;
            }
            this._queue.push(float32);
            this._queued += float32.length;
        };
    }

    /**
     * Called by the audio engine every render quantum (128 samples at 16 kHz
     * = 8 ms).  Must never throw; runs on a high-priority real-time thread.
     */
    process(_inputs, outputs) {
        const output = outputs[0][0];     // mono output channel
        const needed = output.length;     // always 128 in the Web Audio spec

        let written = 0;

        while (written < needed && this._queue.length > 0) {
            const head = this._queue[0];
            const available = head.length;
            const toCopy = Math.min(needed - written, available);

            output.set(head.subarray(0, toCopy), written);
            written += toCopy;

            if (toCopy < available) {
                // Partial consume – trim the head chunk.
                this._queue[0] = head.subarray(toCopy);
            } else {
                // Fully consumed – remove the chunk.
                this._queue.shift();
            }
            this._queued -= toCopy;
        }

        // Fill any remaining output with silence (buffer underrun).
        if (written < needed) {
            output.fill(0, written);
        }

        return true;    // keep the processor alive
    }
}

registerProcessor('pcm-player-processor', PCMPlayerProcessor);
