// 테스트 편의를 위해 Warlus 연동 코드는 삭제

import 'dotenv/config'
import { getFullnodeUrl, SuiClient } from '@mysten/sui/client';
import { MIST_PER_SUI } from '@mysten/sui/utils';
import { Transaction } from '@mysten/sui/transactions';
import { Ed25519Keypair } from '@mysten/sui/keypairs/ed25519';
import { fromHex } from '@mysten/bcs';
import promptSync from 'prompt-sync';
import ora from "ora";
import fs from 'fs';


const network = process.env.NETWORK;
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const TENSROFLOW_SUI_PACKAGE_ID = process.env.PACKAGE_ID;

if (!PRIVATE_KEY) {
	console.error("=============================================");
	console.error("Please provide a PRIVATE_KEY in .env file");
	console.error("=============================================");
	process.exit(1); 
}

const prompt = promptSync();

let input_mag = [];
let input_sign = [];

const rpcUrl = getFullnodeUrl(network);
const client = new SuiClient({ url: rpcUrl });

let SignedFixedGraph = "";
let PartialDenses = "";
let totalGasUsage = 0;

async function getInput(label) {
	try {
		// input data를 Warlus가 아니라 로컬 json 파일에서 읽어도록 수정
		/*
		const response = await fetch('http://localhost:8083/get', {
			method: 'POST', 
			headers: { 'Content-Type': 'application/json' }, 
			body: JSON.stringify({ label: label }) // JSON 데이터 변환
		  });

		const data = await response.json();
		*/
		const fileName = `input_${label}.json`;
		const raw = fs.readFileSync(fileName, 'utf8');
		const data = JSON.parse(raw);

		console.log("Local data: ", data);
		return data;
	} catch (error) {
		console.error('API Call Err:', error)
		return "";
	}
}

/*
async function store(digest_arr, partialDenses_digest_arr, version_arr) {
}
*/
 
async function run() {

	let keypair;
	let result;

	let tx_digest_arr = [];
	let partialDenses_digest_arr = [];
	let version_arr = [];

	while (true) {
		console.log("\n  ");
		const command = prompt(">> Please enter your command : ");

		switch (command.trim().toLowerCase()) {

		case "help":

		console.log(`
\x1b[38;5;199m1. Initialize (init)\x1b[0m
\x1b[38;5;147m- Publishes objects from the Move package
- Creates two main objects:\x1b[0m
    \x1b[38;5;226ma) SignedFixedGraph:\x1b[0m Contains all graph information from the published web3 model
        \x1b[38;5;251m• Weights, biases, and network architecture
        • Immutable after initialization\x1b[0m
    \x1b[38;5;226mb) PartialDenses:\x1b[0m Stores computation results of nodes
        \x1b[38;5;251m• Used for split transaction computation
        • Mutable state for intermediate results\x1b[0m`);

		await sleep(500);

		console.log(`
\x1b[38;5;199m2. Load Input (load input)\x1b[0m
\x1b[38;5;147m- Fetches input data from Walrus blob storage
- Blob contains model inputs uploaded by model publisher
- Prepares input vectors for inference:\x1b[0m
    \x1b[38;5;251m• input_mag: Magnitude vector
    • input_sign: Sign vector\x1b[0m`);

		await sleep(500);

		console.log(`
\x1b[38;5;199m3. Run Inference (run)\x1b[0m
\x1b[38;5;147mThe inference process combines two optimization strategies:\x1b[0m

\x1b[38;5;226mA. Split Transaction Computation (16 parts)\x1b[0m
    \x1b[38;5;251m- Breaks down input layer → hidden layer computation
    - Input (#49 nodes) → Hidden Layer 1 (#16 nodes)
    - Processes in 16 separate transactions for gas efficiency\x1b[0m

\x1b[38;5;226mB. PTB (Parallel Transaction Blocks)\x1b[0m
    \x1b[38;5;251m- Handles remaining layers atomically
    - Hidden Layer 1 → Hidden Layer 2 → Output
    - Executes final classification in single transaction
    - Ensures atomic state transitions\x1b[0m`);

		await sleep(500);

		console.log(`
\x1b[38;5;199m4. Save Receipt to Walrus\x1b[0m
\x1b[38;5;147m- Packages inference evidence:\x1b[0m
    \x1b[38;5;251m• Transaction digests (tx_digest_arr)
    • Partial dense computation proofs (partialDenses_digest_arr)
    • State versions (version_arr)\x1b[0m
\x1b[38;5;147m- Uploads to Walrus as receipt
- Provides permanent proof of inference execution
- Enables digital provenance verification
- Returns Walrus blob ID for reference\x1b[0m`);

		await sleep(500);
			break;
			
		case "init":
			console.log("\nInitializing... \n");
console.log(`
\x1b[38;5;199m1. Initialize (init)\x1b[0m
\x1b[38;5;147m- Publishes objects from the Move package
- Creates two main objects:\x1b[0m
\x1b[38;5;226ma) SignedFixedGraph:\x1b[0m Contains all graph information from the published web3 model
\x1b[38;5;251m• Weights, biases, and network architecture
• Immutable after initialization\x1b[0m
\x1b[38;5;226mb) PartialDenses:\x1b[0m Stores computation results of nodes
\x1b[38;5;251m• Used for split transaction computation
• Mutable state for intermediate results\x1b[0m`);

			let tx = new Transaction();

			if (!tx.gas) {
				console.error("Gas object is not set correctly");
			}

			tx.moveCall({
				target: `${TENSROFLOW_SUI_PACKAGE_ID}::model::initialize`,
			})

			keypair = Ed25519Keypair.fromSecretKey(fromHex(PRIVATE_KEY));
			result = await client.signAndExecuteTransaction({
				transaction: tx,
				signer: keypair,
				options: {
					showEffects: true,
					showEvents: true,
					showObjectChanges: true,
				}
			})

			for (let i=0; i < result['objectChanges'].length; i++) {

				let parts;
				let exist;
				
				parts = result['objectChanges'][i]["objectType"].split("::");
				exist = parts.some(part => part.includes("PartialDenses"));
				if (exist == true) {
					PartialDenses = result['objectChanges'][i]["objectId"];
					exist = false;
				}

				parts = result['objectChanges'][i]["objectType"].split("::");
				exist = parts.some(part => part.includes("SignedFixedGraph"));
				if (exist == true) {
					SignedFixedGraph = result['objectChanges'][i]["objectId"];
					exist = false;
				}
			}
			console.log("");
			console.log("");

			console.log("SignedFixedGraph:", SignedFixedGraph);
			console.log("https://suiscan.xyz/"+network+"/object/"+ SignedFixedGraph+"/tx-blocks");

			console.log("PartialVariable:", PartialDenses);
			console.log("https://suiscan.xyz/"+network+"/object/"+ PartialDenses+"/tx-blocks");

			console.log("Gas Used (only once):", (Number(result.effects.gasUsed.computationCost) + Number(result.effects.gasUsed.storageCost) + Number(result.effects.gasUsed.storageRebate)) / Number(MIST_PER_SUI), " SUI");
			console.log("");

			console.log("");

			console.log(`

				\x1b[38;5;51m╔════════════════════════════════════════════════════════════╗
				║  Completed! init the model  ║ "load input" to load input data from Walrus
				╚════════════════════════════════════════════════════════════╝\x1b[0m
				`);

			break;
			
		case "load input":
console.log(`
\x1b[38;5;199m2. Load Input (load input)\x1b[0m
\x1b[38;5;147m- Fetches input data from Walrus blob storage
- Blob contains model inputs uploaded by model publisher
- Prepares input vectors for inference:\x1b[0m
\x1b[38;5;251m• input_mag: Magnitude vector
• input_sign: Sign vector\x1b[0m`);

			const label = prompt(">> What label do you want? ");

			console.log(label)

			let input = await getInput(Number(label));
			input_mag = input["inputMag"];
			input_sign = input["inputSign"];

			console.log(`

				\x1b[38;5;51m╔════════════════════════════════════════════════════════════╗
				║ Completed! load input data  ║ "run" to start inference
				╚════════════════════════════════════════════════════════════╝\x1b[0m
				`);
			
			break;

		case "run":
console.log(`
\x1b[38;5;199m3. Run Inference (run)\x1b[0m
\x1b[38;5;147mThe inference process combines two optimization strategies:\x1b[0m

\x1b[38;5;226mA. Split Transaction Computation (16 parts)\x1b[0m
\x1b[38;5;251m- Breaks down input layer → hidden layer computation
- Input (#49 nodes) → Hidden Layer 1 (#16 nodes)
- Processes in 16 separate transactions for gas efficiency\x1b[0m

\x1b[38;5;226mB. PTB (Parallel Transaction Blocks)\x1b[0m
\x1b[38;5;251m- Handles remaining layers atomically
- Hidden Layer 1 → Hidden Layer 2 → Output
- Executes final classification in single transaction
- Ensures atomic state transitions\x1b[0m`);

			console.log('\nInference start... \n');

			let totalTasks = 2
			let spinner;
			
			for (let i = 0; i<totalTasks; i++) {

				await sleep(500);

				const filledBar = '█'.repeat(i+1);  
				const emptyBar = '░'.repeat(totalTasks - i - 1); 
				const progressBar = filledBar + emptyBar; // total progress bar
				
				if (i == totalTasks-1) {

					let final_tx = new Transaction();

					if (!final_tx.gas) {
						console.error("Gas object is not set correctly");
					}

					let res_act1 = final_tx.moveCall({
						target: `${TENSROFLOW_SUI_PACKAGE_ID}::graph::split_chunk_finalize`,
						arguments: [
							final_tx.object(PartialDenses),
							//final_tx.pure.string('dense'),
							final_tx.pure.vector('u8', [100, 101, 110, 115, 101]),
						],
					})

					let res_act2 = final_tx.moveCall({
						target: `${TENSROFLOW_SUI_PACKAGE_ID}::graph::ptb_layer`,
						arguments: [
							final_tx.object(SignedFixedGraph),
							res_act1[0],
							res_act1[1],
							res_act1[2],
							//final_tx.pure.string('dense_1'),
							final_tx.pure.vector('u8', [100, 101, 110, 115, 101, 95, 49]),
						],
					})

					final_tx.moveCall({
						target: `${TENSROFLOW_SUI_PACKAGE_ID}::graph::ptb_layer_arg_max`,
						arguments: [
							final_tx.object(SignedFixedGraph),
							res_act2[0],
							res_act2[1],
							res_act2[2],
							//final_tx.pure.string('dense_2'),
							final_tx.pure.vector('u8', [100, 101, 110, 115, 101, 95, 50]),
						],
					})

					keypair = Ed25519Keypair.fromSecretKey(fromHex(PRIVATE_KEY));
					result = await client.signAndExecuteTransaction({
						transaction: final_tx,
						signer: keypair,
						options: {
							showEffects: true,
							showEvents: true,
							showObjectChanges: true,
						}
					})
					spinner.succeed("✅ spilit transaction computation completed!");
					
					spinner = ora("Processing task... ").start();
					console.log("\n***** Start PTB computation hidden layer 1 -> hidden layer 2 -> output *****");

					for (let i=0; i < result['objectChanges'].length; i++) {

						let parts;
						let exist;
						
						parts = result['objectChanges'][i]["objectType"].split("::");
						exist = parts.some(part => part.includes("PartialDenses"));
						if (exist == true) {
							partialDenses_digest_arr.push(result['objectChanges'][i]["digest"]);
							version_arr.push(result['objectChanges'][i]["version"]);
						}
					}
					
					tx_digest_arr.push(result.digest)
					console.log("\nTx Digest:", result.digest)
					console.log("https://suiscan.xyz/"+network+"/tx/"+ result.digest);

					console.log(`
					PTB Transaction (remain other layers and activation (ReLU))
					┌───────────────────────────────────────────────────────────────────┐
					│                                               		    │
					│   Layer 1(16) →→→ Layer 2 (8) →→→ Layer 3 (10) →→→ argmax (1)	    │
					│                                               		    │
					└───────────────────────────────────────────────────────────────────┘
					`);

					console.log("Gas Used: ", (Number(result.effects.gasUsed.computationCost) + Number(result.effects.gasUsed.nonRefundableStorageFee)) / Number(MIST_PER_SUI), " SUI");
					totalGasUsage += Number(result.effects.gasUsed.computationCost) + Number(result.effects.gasUsed.nonRefundableStorageFee)

					console.log("\nresult:", result);
					console.log("\n");
					console.log("\nresult:", result.events[0].parsedJson['value']);
					console.log("Total Gas Used (SUI):", totalGasUsage / Number(MIST_PER_SUI))
					spinner.succeed("✅ PTB transaction computation completed!");

					//const data = await store(tx_digest_arr, partialDenses_digest_arr, version_arr);

					totalGasUsage = 0;
					tx_digest_arr = [];
					partialDenses_digest_arr = [];
					version_arr = [];
				} else {

					let tx = new Transaction();

					if (!tx.gas) {
						console.error("Gas object is not set correctly");
					}
					console.log(`input layer  -> hidden layer 1 ${i+1}/${totalTasks-1}`);
					
					// Add visualization of split computation
					console.log("\nSplit Computation Progress:");
					console.log(`
    Split Transaction ${(i+1).toString().padStart(2, ' ')}/16 
    ┌───────────────────────────────────────────────┐
    │                                               │
    │   Input Layer (49) →→→ Layer 1 Node ${(i+1).toString().padStart(2, ' ')}/16     │
    │                                               │
    │   Computing weights & biases contribution     │
    │   from all input nodes to L1 hidden node ${(i+1).toString().padStart(2, ' ')}   │
    │                                               │
    └───────────────────────────────────────────────┘
`);

					tx.moveCall({
						target: `${TENSROFLOW_SUI_PACKAGE_ID}::graph::split_chunk_compute`,
						arguments: [
							tx.object(SignedFixedGraph),
							tx.object(PartialDenses),
							tx.pure.string('dense'),
							tx.pure.vector('u64', input_mag),
							tx.pure.vector('u64', input_sign),
							tx.pure.u64(1),
							tx.pure.u64(i),
							tx.pure.u64(i),
						],
					})

					keypair = Ed25519Keypair.fromSecretKey(fromHex(PRIVATE_KEY));
					result = await client.signAndExecuteTransaction({
						transaction: tx,
						signer: keypair,
						options: {
							showEffects: true,
							showEvents: true,
							showObjectChanges: true,
						}
					})

					spinner = ora("Processing task... ").start();
					console.log(progressBar + ` ${i+1}/${totalTasks}`);
				
					for (let i=0; i < result['objectChanges'].length; i++) {

						let parts;
						let exist;
						
						parts = result['objectChanges'][i]["objectType"].split("::");
						exist = parts.some(part => part.includes("PartialDenses"));
						if (exist == true) {
							partialDenses_digest_arr.push(result['objectChanges'][i]["digest"]);
							version_arr.push(result['objectChanges'][i]["version"]);
						}
					}
					
					tx_digest_arr.push(result.digest)
					console.log("Tx Digest:", result.digest)
					console.log("https://suiscan.xyz/"+network+"/tx/"+ result.digest);

					console.log("Gas Used:", (Number(result.effects.gasUsed.computationCost) + Number(result.effects.gasUsed.nonRefundableStorageFee)) / Number(MIST_PER_SUI) , " SUI"); 
					console.log("");
					totalGasUsage += Number(result.effects.gasUsed.computationCost) + Number(result.effects.gasUsed.nonRefundableStorageFee)
				}
			}
			console.log(`

				\x1b[38;5;51m╔════════════════════════════════════════════════════════════╗
				║  inference completed! ║ "load input" to load input data from Walrus to next inference
				╚════════════════════════════════════════════════════════════╝\x1b[0m
				`);

			break;
			
		default:
			console.log(`Unknown command: '${command}'`);
		}
	}
  }

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function displayLog() {
    console.log(`
\x1b[38;5;51m╔════════════════════════════════════════════════════════════╗
║  \x1b[38;5;213mOPENGRAPH: Fully On-chain Neural Network Inference\x1b[38;5;51m        ║ 
╚════════════════════════════════════════════════════════════╝\x1b[0m`);
console.log("\n 'help' for more commands");

console.log(`
\x1b[38;5;51m╔════════════════════════════════════════════════════════════╗
║                Ready to start inference!                   ║ "init" to initialize the model
╚════════════════════════════════════════════════════════════╝\x1b[0m
`);
}

// Call the async function
displayLog();
run();

  





