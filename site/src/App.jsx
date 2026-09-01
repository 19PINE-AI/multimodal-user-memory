import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { PaperOverview } from './components/PaperOverview';
import { Reproducibility } from './components/Reproducibility';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen bg-paper text-ink">
      <Navbar />
      <main>
        <Hero />
        <PaperOverview />
        <Reproducibility />
      </main>
      <Footer />
    </div>
  );
}
