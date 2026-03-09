package AkoireHyMeKo_Architecture {

    doc /* * The formal specification of the AKOIRE-HyMeKo neuro-symbolic system.
     * Maps probabilistic cognitive intent to deterministic hypergraph tensors.
     */

    // --- Part Definitions ---

    part def CognitiveOperatingSystem {
        in port currentAmbience;
        in port externalIntent;
        in port systemObjectives;
        in port physicsKyosei;

        out port proposedRefinement; // Outputs LALR-compatible string

        action synthesizeRefinement {
            in ambience = currentAmbience;
            in intent = externalIntent;
            in constraints = physicsKyosei;
            out mutationString = proposedRefinement;
        }
    }

    part def DeterministicHypergraphEngine {
        in port incomingRefinement;
        out port newAmbience;
        out port errorFeedback;

        action evaluateSyntax {
            doc /* LALRPOP Parser Gatekeeper */
            in stringInput = incomingRefinement;
            out validAST;
            out parseError;
        }

        action mutateActiveState {
            doc /* O(1) Push to TensorCoo */
            in ast = evaluateSyntax.validAST;
            out activeCooState;
        }

        action freezeState {
            doc /* Compress to TensorCsr for Graph Traversal */
            in coo = mutateActiveState.activeCooState;
            out compressedCsr = newAmbience;
        }
    }

    // --- System Topology ---

    part akoire_hymeko_system {
        part akoire: CognitiveOperatingSystem;
        part hymeko: DeterministicHypergraphEngine;

        // The Synthesis Flow
        flow akoire.proposedRefinement to hymeko.incomingRefinement;

        // The Success Loop (New state feeds back to Ambience)
        flow hymeko.newAmbience to akoire.currentAmbience;

        // The Error Loop (Syntax failures force self-correction)
        flow hymeko.errorFeedback to akoire.currentAmbience;
    }
}