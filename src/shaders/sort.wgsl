const RADIX_BASE = 256u;
const RADIX_BITS_PER_DIGIT = 8u;
const RADIX_DIGIT_PLACES = 4u;
const ENTRIES_PER_INVOCATION = 4u;
const WORKGROUP_ENTRIES_BINNING = RADIX_BASE * ENTRIES_PER_INVOCATION;

struct DrawIndirect {
	vertexCount: u32,
	instanceCount: atomic<u32>,
	_baseVertex: u32,
	_baseInstance: u32,
};

struct SortingGlobal {
	statusCounter: array<array<atomic<u32>, RADIX_BASE>, MAX_TILE_COUNT_BINNING>,
	digitHistogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>,
	drawIndirect: DrawIndirect,
	assignmentCounter: atomic<u32>,
};

@group(0) @binding(2) var<uniform> passIndex: u32;
@group(0) @binding(3) var<storage, read_write> sorting: SortingGlobal;
@group(0) @binding(4) var<storage, read_write> entriesIn: array<Entry>;
@group(0) @binding(5) var<storage, read_write> entriesOut: array<Entry>;

// Onesweep Radix Sort
struct LocalHistogram {
	digitHistogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>,
}
var<workgroup> localHistogram: LocalHistogram;

@compute @workgroup_size(RADIX_BASE, RADIX_DIGIT_PLACES)
fn histogram(
	@builtin(local_invocation_id) localID: vec3<u32>,
	@builtin(global_invocation_id) globalID: vec3<u32>,
) {
	atomicStore(&localHistogram.digitHistogram[localID.y][localID.x], 0u);
	workgroupBarrier();

	let thread_index = globalID.x * RADIX_DIGIT_PLACES + globalID.y;
	let start_entry_index = thread_index * ENTRIES_PER_INVOCATION;
	let end_entry_index = start_entry_index + ENTRIES_PER_INVOCATION;
	for(var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
		let splat_index = entry_index * stride;
		if(splat_index >= arrayLength(&gaussianData)) { continue; }

		var key: u32 = 0xFFFFFFFFu; // Stream compaction for frustum culling
		let pos = readVec3(splat_index + posOffset);
		var clipPos = cam.projection * cam.view * vec4f(pos, 1);
		clipPos /= clipPos.w;
		if(isInFrustum(clipPos.xyz)) {
			key = bitcast<u32>(1 - clipPos.z);
            // key |= u32((clipPos.x * 0.5 + 0.5) * 0xFF.0) << 8u;
            // key |= u32((clipPos.y * 0.5 + 0.5) * 0xFF.0);
		}

		entriesOut[entry_index].key = key;
		entriesOut[entry_index].value = entry_index;
		for(var shift = 0u; shift < RADIX_DIGIT_PLACES; shift += 1u) {
			let digit = (key >> (shift * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
			atomicAdd(&localHistogram.digitHistogram[shift][digit], 1u);
		}
	}
	workgroupBarrier();

	let local = atomicLoad(&localHistogram.digitHistogram[localID.y][localID.x]);
	atomicAdd(&sorting.digitHistogram[localID.y][localID.x], local);
}

@compute @workgroup_size(1)
fn prefixSum(@builtin(global_invocation_id) globalID: vec3<u32>) {
	var sum = 0u;
	for(var digit = 0u; digit < RADIX_BASE; digit += 1u) {
		let tmp = atomicLoad(&sorting.digitHistogram[globalID.y][digit]);
		atomicStore(&sorting.digitHistogram[globalID.y][digit], sum);
		sum += tmp;
	}
}

struct LocalBinning {
	entries: array<atomic<u32>, WORKGROUP_ENTRIES_BINNING>,
	gatherSources: array<atomic<u32>, WORKGROUP_ENTRIES_BINNING>,
	scan: array<atomic<u32>, RADIX_BASE>,
	total: u32,
}

var<workgroup> localBinning: LocalBinning;

fn exclusiveScan(index: u32, value: u32) -> u32 {
	atomicStore(&localBinning.scan[index], value);
	var offset = 1u;
	for(var d = RADIX_BASE >> 1u; d > 0u; d >>= 1u) {
		workgroupBarrier();
		if(index < d) {
			var ai = offset * (2u * index + 1u) - 1u;
			var bi = offset * (2u * index + 2u) - 1u;
			let a = atomicLoad(&localBinning.scan[ai]);
			atomicAdd(&localBinning.scan[bi], a);
		}
		offset <<= 1u;
	}
	if(index == 0u) {
	  var i = RADIX_BASE - 1u;
	  localBinning.total = atomicExchange(&localBinning.scan[i], 0u);
	}
	for(var d = 1u; d < RADIX_BASE; d <<= 1u) {
		workgroupBarrier();
		offset >>= 1u;
		if(index < d) {
			var ai = offset * (2u * index + 1u) - 1u;
			var bi = offset * (2u * index + 2u) - 1u;
			let t = atomicLoad(&localBinning.scan[ai]);
			let b = atomicAdd(&localBinning.scan[bi], t);
			atomicStore(&localBinning.scan[ai], b);
		}
	}
	workgroupBarrier();
	return atomicLoad(&localBinning.scan[index]);
}

@compute @workgroup_size(RADIX_BASE)
fn bin(
	@builtin(local_invocation_id) localID: vec3<u32>,
	@builtin(global_invocation_id) globalID: vec3<u32>,
) {
	// Draw an assignment number
	if(localID.x == 0u) {
		atomicStore(&localBinning.entries[0], atomicAdd(&sorting.assignmentCounter, 1u));
	}
	// Reset histogram
	atomicStore(&localBinning.scan[localID.x], 0u);
	workgroupBarrier();

	let assignment = atomicLoad(&localBinning.entries[0]);
	let global_entry_offset = assignment * WORKGROUP_ENTRIES_BINNING;
	// TODO: Specialize end shader
	if(localID.x == 0u && (assignment * WORKGROUP_ENTRIES_BINNING + WORKGROUP_ENTRIES_BINNING) * stride >= arrayLength(&gaussianData)) {
		// Last workgroup resets the assignment number for the next pass
		atomicStore(&sorting.assignmentCounter, 0u);
	}

	// Load keys from global memory into registers and rank them
	var keys: array<u32, ENTRIES_PER_INVOCATION>;
	var ranks: array<u32, ENTRIES_PER_INVOCATION>;
	for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		keys[entry_index] = entriesIn[global_entry_offset + RADIX_BASE * entry_index + localID.x].key;
		let digit = (keys[entry_index] >> (passIndex * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
		ranks[entry_index] = atomicAdd(&localBinning.scan[digit], 1u);
	}
	workgroupBarrier();

	// Cumulate histogram
	let local_digit_count = atomicLoad(&localBinning.scan[localID.x]);
	let local_digit_offset = exclusiveScan(localID.x, local_digit_count);
	atomicStore(&localBinning.scan[localID.x], local_digit_offset);

	// Chained decoupling lookback
	atomicStore(&sorting.statusCounter[assignment][localID.x], 0x40000000u | local_digit_count);
	var global_digit_count = 0u;
	var previous_tile = assignment;
	while true {
		if(previous_tile == 0u) {
			global_digit_count += atomicLoad(&sorting.digitHistogram[passIndex][localID.x]);
			break;
		}
		previous_tile -= 1u;
		var status_counter = 0u;
		while((status_counter & 0xC0000000u) == 0u) {
			status_counter = atomicLoad(&sorting.statusCounter[previous_tile][localID.x]);
		}
		global_digit_count += status_counter & 0x3FFFFFFFu;
		if((status_counter & 0x80000000u) != 0u) {
			break;
		}
	}

	atomicStore(&sorting.statusCounter[assignment][localID.x], 0x80000000u | (global_digit_count + local_digit_count));
	if(passIndex == RADIX_DIGIT_PLACES - 1u && localID.x == RADIX_BASE - 2u && (global_entry_offset + WORKGROUP_ENTRIES_BINNING) * stride >= arrayLength(&gaussianData)) {
		sorting.drawIndirect.vertexCount = 4u;
		atomicStore(&sorting.drawIndirect.instanceCount, global_digit_count + local_digit_count);
	}

	// Scatter keys inside shared memory
	for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let key = keys[entry_index];
		let digit = (key >> (passIndex * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
		ranks[entry_index] += atomicLoad(&localBinning.scan[digit]);
		atomicStore(&localBinning.entries[ranks[entry_index]], key);
	}
	workgroupBarrier();

	// Add global offset
	atomicStore(&localBinning.scan[localID.x], global_digit_count - local_digit_offset);
	workgroupBarrier();

	// Store keys from shared memory into global memory
	for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let key = atomicLoad(&localBinning.entries[RADIX_BASE * entry_index + localID.x]);
		let digit = (key >> (passIndex * RADIX_BITS_PER_DIGIT)) & (RADIX_BASE - 1u);
		keys[entry_index] = digit;
		entriesOut[atomicLoad(&localBinning.scan[digit]) + RADIX_BASE * entry_index + localID.x].key = key;
	}
	workgroupBarrier();

	// Load values from global memory and scatter them inside shared memory
	for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let value = entriesIn[global_entry_offset + RADIX_BASE * entry_index + localID.x].value;
		atomicStore(&localBinning.entries[ranks[entry_index]], value);
	}
	workgroupBarrier();

	// Store values from shared memory into global memory
	for(var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let value = atomicLoad(&localBinning.entries[RADIX_BASE * entry_index + localID.x]);
		let digit = keys[entry_index];
		entriesOut[atomicLoad(&localBinning.scan[digit]) + RADIX_BASE * entry_index + localID.x].value = value;
	}
}